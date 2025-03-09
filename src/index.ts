#!/usr/bin/env node

/**
 * MCP server that implements RAG functionality for documents in ~/docs directory.
 * Features:
 * - List available documents as resources
 * - Read and query documents using RAG
 * - Add new documents via git clone or wget
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListResourcesRequestSchema,
  ListToolsRequestSchema,
  ReadResourceRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { exec } from "child_process";
import { promisify } from "util";
import {
  VectorStoreIndex,
  Document,
  storageContextFromDefaults,
  SimpleVectorStore,
  CallbackManager,
  Settings
} from "llamaindex";
import { Gemini, GEMINI_MODEL, GeminiEmbedding } from "@llamaindex/google";

const execAsync = promisify(exec);

// GEMINI_API_KEYをGOOGLE_API_KEYにマッピング（ライブラリが自動的に取得するため）
if (process.env.GEMINI_API_KEY && !process.env.GOOGLE_API_KEY) {
  process.env.GOOGLE_API_KEY = process.env.GEMINI_API_KEY;
}

// Get docs path from environment variable, with fallback
const DOCS_PATH = process.env.DOCS_PATH || path.join(process.env.HOME || process.env.USERPROFILE || '', 'docs');

// Ensure docs directory exists
if (!fs.existsSync(DOCS_PATH)) {
  fs.mkdirSync(DOCS_PATH, { recursive: true });
}

// Ensure .indices directory exists
const INDICES_PATH = path.join(DOCS_PATH, '.indices');
if (!fs.existsSync(INDICES_PATH)) {
  fs.mkdirSync(INDICES_PATH, { recursive: true });
}

// Store indices for each document
const indices: Record<string, { index: VectorStoreIndex, description: string }> = {};

/**
 * Normalizes a repository name from its URL or path
 */
function normalizeRepoName(repoUrl: string): string {
  const parts = repoUrl.split('/');
  return parts[parts.length - 1].replace('.git', '');
}

/**
 * Lists all available documents in the docs directory
 */
async function listDocuments(): Promise<Array<{ id: string, name: string, path: string, description: string }>> {
  const documents: Array<{ id: string, name: string, path: string, description: string }> = [];
  
  const entries = fs.readdirSync(DOCS_PATH, { withFileTypes: true });
  
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    
    const entryPath = path.join(DOCS_PATH, entry.name);
    
    // Gitリポジトリかテキストファイルかを判断
    let isGitRepo = false;
    try {
      // .gitディレクトリが存在するか確認
      const gitPath = path.join(entryPath, '.git');
      isGitRepo = fs.existsSync(gitPath) && fs.statSync(gitPath).isDirectory();
    } catch (error) {
      // エラーが発生した場合はGitリポジトリではない
      isGitRepo = false;
    }
    
    // index.txtファイルがあるか確認
    const indexPath = path.join(entryPath, 'index.txt');
    const hasIndexFile = fs.existsSync(indexPath) && fs.statSync(indexPath).isFile();
    
    if (isGitRepo) {
      documents.push({
        id: entry.name,
        name: entry.name,
        path: entryPath,
        description: `Git repository: ${entry.name}`,
      });
    } else if (hasIndexFile) {
      documents.push({
        id: entry.name,
        name: entry.name,
        path: indexPath,
        description: `Text document: ${entry.name}`,
      });
    }
  }
  
  return documents;
}

/**
 * ディレクトリ内のファイルを再帰的に読み込む関数
 * @param dirPath 対象ディレクトリのパス
 * @returns ドキュメントの配列
 */
async function readDirectoryRecursively(dirPath: string): Promise<Document[]> {
  const result: Document[] = [];
  const entries = fs.readdirSync(dirPath, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    
    if (entry.isDirectory()) {
      // サブディレクトリを再帰的に処理
      const subdirDocs = await readDirectoryRecursively(fullPath);
      result.push(...subdirDocs);
    } else if (entry.isFile() && !entry.name.startsWith('.')) {
      // ファイルを処理
      try {
        const content = fs.readFileSync(fullPath, 'utf-8');
        // ファイルの相対パスを保存（DOCS_PATHからの相対パス）
        const relativePath = path.relative(DOCS_PATH, fullPath);
        result.push(new Document({ 
          text: content, 
          metadata: { 
            name: entry.name, 
            source: fullPath,
            path: relativePath
          } 
        }));
      } catch (error: any) {
        console.warn(`Failed to read file ${fullPath}: ${error.message}`);
      }
    }
  }
  
  return result;
}

/**
 * Load and index a document
 * 存在しないドキュメントの場合は自動的に作成を試みる
 */
async function loadDocument(documentId: string): Promise<VectorStoreIndex> {
  if (indices[documentId]?.index) {
    return indices[documentId].index;
  }

  let documents = await listDocuments();
  let document = documents.find(c => c.id === documentId);
  
  // ドキュメントが存在しない場合はエラーをスロー
  if (!document) {
    throw new Error(`Document not found: ${documentId}`);
  }

  let documentItems: Document[] = [];
  
  if (fs.statSync(document.path).isDirectory()) {
    // ディレクトリを再帰的に処理
    documentItems = await readDirectoryRecursively(document.path);
    
    // 空のドキュメントリストの場合にフォールバックメッセージを追加
    if (documentItems.length === 0) {
      console.warn(`No documents found in document: ${documentId}`);
      documentItems.push(new Document({ 
        text: `This document (${document.name}) appears to be empty. Please check if files exist at path: ${document.path}`, 
        metadata: { name: 'empty-notice', source: document.path } 
      }));
    }
  } else {
    // Process single file
    const text = fs.readFileSync(document.path, 'utf-8');
    documentItems = [new Document({ text, metadata: { name: document.id, source: document.path } })];
  }
  
  // 一時的にGemini埋め込みモデルを設定
  const originalEmbedModel = Settings.embedModel;
  // GeminiEmbeddingはデフォルトでgemini-proモデルを使用
  const geminiEmbed = new GeminiEmbedding();
  
  // グローバル設定に設定
  Settings.embedModel = geminiEmbed;
  
  // Create storage context
  const storageContext = await storageContextFromDefaults({
    persistDir: path.join(DOCS_PATH, '.indices', documentId),
  });
  
  // Create index
  const index = await VectorStoreIndex.fromDocuments(documentItems, {
    storageContext,
  });
  
  // 元の設定に戻す
  Settings.embedModel = originalEmbedModel;
  
  // Save index for future use
  indices[documentId] = { 
    index, 
    description: document.description 
  };
  
  return index;
}

/**
 * Clone a git repository to the docs directory
 * @param repoUrl URL of the Git repository to clone
 * @param subdirectory Optional specific subdirectory to sparse checkout
 * @param documentName Optional custom name for the document
 */
async function cloneRepository(repoUrl: string, subdirectory?: string, documentName?: string): Promise<string> {
  // Use custom document name if provided, otherwise normalize repo name
  const repoName = documentName || normalizeRepoName(repoUrl);
  const repoPath = path.join(DOCS_PATH, repoName);
  
  // Check if repository already exists
  if (fs.existsSync(repoPath)) {
    // Pull latest changes
    await execAsync(`cd "${repoPath}" && git pull`);
    
    // If subdirectory is specified, make sure it's in the sparse-checkout
    if (subdirectory) {
      await execAsync(`cd "${repoPath}" && git sparse-checkout set ${subdirectory}`);
    }
  } else {
    if (subdirectory) {
      // Clone with sparse-checkout for specific subdirectory
      await execAsync(`mkdir -p "${repoPath}" && cd "${repoPath}" && \
                      git init && \
                      git remote add origin ${repoUrl} && \
                      git config core.sparseCheckout true && \
                      git config --local core.autocrlf false && \
                      echo "${subdirectory}/*" >> .git/info/sparse-checkout && \
                      git pull --depth=1 origin main || git pull --depth=1 origin master`);
    } else {
      // Normal clone for the entire repository
      await execAsync(`cd "${DOCS_PATH}" && git clone ${repoUrl}`);
    }
  }
  
  return repoName;
}

/**
 * Download a file from URL to the docs directory
 * @param fileUrl ダウンロードするファイルのURL
 * @param documentName ドキュメント名（ディレクトリ名として使用）
 */
async function downloadFile(fileUrl: string, documentName: string): Promise<string> {
  // ドキュメント用のディレクトリを作成
  const docDir = path.join(DOCS_PATH, documentName);
  
  // ディレクトリが存在しない場合は作成
  if (!fs.existsSync(docDir)) {
    fs.mkdirSync(docDir, { recursive: true });
  }
  
  // ファイル名を取得（URLのパス部分の最後）
  const fileName = path.basename(fileUrl);
  
  // index.txtとしてファイルを保存
  const filePath = path.join(docDir, 'index.txt');
  
  // ファイルをダウンロード
  await execAsync(`cd "${docDir}" && wget -O "index.txt" ${fileUrl}`);
  
  return documentName;
}

/**
 * Create MCP server
 */
const server = new Server(
  {
    name: "mcp-docs-rag",
    version: "0.1.0",
  },
  {
    capabilities: {
      resources: {},
      tools: {},
      prompts: {},
    },
  }
);

/**
 * Handler for listing available documents as resources
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  const documents = await listDocuments();
  
  return {
    resources: documents.map(document => ({
      uri: `docs:///${document.id}`,
      mimeType: "text/plain",
      name: document.name,
      description: document.description
    }))
  };
});

/**
 * Handler for reading a document
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const url = new URL(request.params.uri);
  const documentId = url.pathname.replace(/^\//, '');
  
  const documents = await listDocuments();
  const document = documents.find(c => c.id === documentId);
  
  if (!document) {
    throw new Error(`Document not found: ${documentId}`);
  }
  
  let content: string;
  
  if (fs.statSync(document.path).isDirectory()) {
    // List files in directory
    const files = fs.readdirSync(document.path, { withFileTypes: true })
      .filter(entry => entry.isFile())
      .map(entry => entry.name);
    
    content = `Repository: ${document.name}\n\nFiles:\n${files.join('\n')}`;
  } else {
    // Read file content
    content = fs.readFileSync(document.path, 'utf-8');
  }
  
  return {
    contents: [{
      uri: request.params.uri,
      mimeType: "text/plain",
      text: content
    }]
  };
});

/**
 * Handler for listing available tools
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_documents",
        description: "List all available documents in the DOCS_PATH directory",
        inputSchema: {
          type: "object",
          properties: {}
        }
      },
      {
        name: "rag_query",
        description: "Query a document using RAG. Note: If the index does not exist, it will be created when you query, which may take some time.",
        inputSchema: {
          type: "object",
          properties: {
            document_id: {
              type: "string",
              description: "ID of the document to query"
            },
            query: {
              type: "string",
              description: "Query to run against the document"
            }
          },
          required: ["document_id", "query"]
        }
      },
      {
        name: "add_git_repository",
        description: "Add a git repository to the docs directory with optional sparse checkout. Please do not use 'docs' in the document name.",
        inputSchema: {
          type: "object",
          properties: {
            repository_url: {
              type: "string",
              description: "URL of the git repository to clone"
            },
            document_name: {
              type: "string",
              description: "Optional: Custom name for the document (defaults to repository name). Use a simple, descriptive name without '-docs' suffix. For example, use 'react' instead of 'react-docs'."
            },
            subdirectory: {
              type: "string",
              description: "Optional: Specific subdirectory to sparse checkout (e.g. 'path/to/specific/dir'). This uses Git's sparse-checkout feature to only download the specified directory."
            }
          },
          required: ["repository_url"]
        }
      },
      {
        name: "add_text_file",
        description: "Add a text file to the docs directory with a specified name. Please do not use 'docs' in the document name.",
        inputSchema: {
          type: "object",
          properties: {
            file_url: {
              type: "string",
              description: "URL of the text file to download"
            },
            document_name: {
              type: "string",
              description: "Name of the document (will be used as directory name). Choose a descriptive name rather than using the URL filename (e.g. 'hono' instead of 'llms-full.txt')"
            }
          },
          required: ["file_url", "document_name"]
        }
      }
    ]
  };
});

/**
 * Handler for tool calls
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  switch (request.params.name) {
    case "list_documents": {
      const documents = await listDocuments();
      
      // ドキュメントの情報を整形
      const documentsList = documents.map(document => {
        return `- ${document.name}: ${document.description}`;
      }).join('\n');
      
      return {
        content: [{
          type: "text",
          text: `Available documents in ${DOCS_PATH}:\n\n${documentsList}\n\nTotal documents: ${documents.length}`
        }]
      };
    }
    
    case "rag_query": {
      const documentId = String(request.params.arguments?.document_id);
      const query = String(request.params.arguments?.query);
      
      if (!documentId || !query) {
        throw new Error("Document ID and query are required");
      }
      
      try {
        // ドキュメントが存在するか確認し、存在しなければ自動的に作成を試みる
        let documents = await listDocuments();
        let document = documents.find(c => c.id === documentId);
        
        if (!document) {
          return {
            content: [{
              type: "text",
              text: `Document '${documentId}' not found. Please add it manually using add_git_repository or add_text_file tools.`
            }]
          };
        }
        
        // Load and index document if needed
        const index = await loadDocument(documentId);
      
      // 一時的にGemini LLMを設定
      const originalLLM = Settings.llm;
      const gemini = new Gemini({
        model: GEMINI_MODEL.GEMINI_2_0_FLASH
      });
      
      // グローバル設定に設定
      Settings.llm = gemini;
      
      // クエリエンジンの作成
      const queryEngine = index.asQueryEngine();
      
      // クエリの実行
      const response = await queryEngine.query({
        query
      });
      
      // 元の設定に戻す
      Settings.llm = originalLLM;
      
      return {
        content: [{
          type: "text",
          text: response.toString()
        }]
      };
      } catch (error: any) {
        console.error(`Error in rag_query:`, error.message);
        return {
          content: [{
            type: "text",
            text: `Error processing query: ${error.message}`
          }]
        };
      }
    }
    
    case "add_git_repository": {
      const repositoryUrl = String(request.params.arguments?.repository_url);
      const subdirectory = request.params.arguments?.subdirectory ? String(request.params.arguments?.subdirectory) : undefined;
      const documentName = request.params.arguments?.document_name ? String(request.params.arguments?.document_name) : undefined;
      
      if (!repositoryUrl) {
        throw new Error("Repository URL is required");
      }
      
      const repoName = await cloneRepository(repositoryUrl, subdirectory, documentName);
      
      // Prepare response message with appropriate details
      let responseText = `Added git repository: ${repoName}`;
      
      if (subdirectory) {
        responseText += ` (sparse checkout of '${subdirectory}')`;  
      }
      
      if (documentName) {
        responseText += ` with custom name '${documentName}'`;  
      }
      
      return {
        content: [{
          type: "text",
          text: `${responseText}. The index will be created when you query this document for the first time.`
        }]
      };
    }
    
    case "add_text_file": {
      const fileUrl = String(request.params.arguments?.file_url);
      const documentName = String(request.params.arguments?.document_name);
      
      if (!fileUrl) {
        throw new Error("File URL is required");
      }
      
      if (!documentName) {
        throw new Error("Document name is required");
      }
      
      const docName = await downloadFile(fileUrl, documentName);
      
      return {
        content: [{
          type: "text",
          text: `Added document '${docName}' with content from ${fileUrl}. The index will be created when you query this document for the first time.`
        }]
      };
    }
    
    default:
      throw new Error("Unknown tool");
  }
});

/**
 * Handler for listing available prompts
 */
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return {
    prompts: [
      {
        name: "guide_documents_usage",
        description: "Guide on how to use documents and RAG functionality",
      }
    ]
  };
});

/**
 * Handler for getting a prompt
 */
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  if (request.params.name !== "guide_documents_usage") {
    throw new Error("Unknown prompt");
  }

  const documents = await listDocuments();
  
  return {
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: "Please list the available documents and guide on how to use the RAG functionality."
        }
      },
      {
        role: "assistant",
        content: {
          type: "text",
          text: `Available documents:\n${documents.map(c => `- ${c.name}: ${c.description}`).join('\n')}\n\nUse the 'rag_query' tool to ask questions about these documents.`
        }
      }
    ]
  };
});

/**
 * Main function to start the server
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

// Start the server
main().catch((error) => {
  console.error("Server error:", error);
  process.exit(1);
});
