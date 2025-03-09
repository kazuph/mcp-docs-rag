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

// Store indices for each document collection
const indices: Record<string, { index: VectorStoreIndex, description: string }> = {};

/**
 * Normalizes a repository name from its URL or path
 */
function normalizeRepoName(repoUrl: string): string {
  const parts = repoUrl.split('/');
  return parts[parts.length - 1].replace('.git', '');
}

/**
 * Lists all available document collections in the docs directory
 */
async function listDocumentCollections(): Promise<Array<{ id: string, name: string, path: string, description: string }>> {
  const collections: Array<{ id: string, name: string, path: string, description: string }> = [];
  
  const entries = fs.readdirSync(DOCS_PATH, { withFileTypes: true });
  
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.name.startsWith('.')) continue;
    
    const entryPath = path.join(DOCS_PATH, entry.name);
    
    // Gitリポジトリかテキストファイルコレクションかを判断
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
      collections.push({
        id: entry.name,
        name: entry.name,
        path: entryPath,
        description: `Git repository: ${entry.name}`,
      });
    } else if (hasIndexFile) {
      collections.push({
        id: entry.name,
        name: entry.name,
        path: indexPath,
        description: `Text document: ${entry.name}`,
      });
    }
  }
  
  return collections;
}

/**
 * Load and index a document collection
 */
async function loadDocumentCollection(collectionId: string): Promise<VectorStoreIndex> {
  if (indices[collectionId]?.index) {
    return indices[collectionId].index;
  }

  const collections = await listDocumentCollections();
  const collection = collections.find(c => c.id === collectionId);
  
  if (!collection) {
    throw new Error(`Document collection not found: ${collectionId}`);
  }

  let documents: Document[] = [];
  
  if (fs.statSync(collection.path).isDirectory()) {
    // Process directory - manually read files
    const files = fs.readdirSync(collection.path, { withFileTypes: true });
    
    for (const file of files) {
      if (file.isFile() && !file.name.startsWith('.')) {
        const filePath = path.join(collection.path, file.name);
        const content = fs.readFileSync(filePath, 'utf-8');
        documents.push(new Document({ text: content, metadata: { name: file.name, source: filePath } }));
      }
    }
  } else {
    // Process single file
    const text = fs.readFileSync(collection.path, 'utf-8');
    documents = [new Document({ text, metadata: { name: collection.id, source: collection.path } })];
  }
  
  // 一時的にGemini埋め込みモデルを設定
  const originalEmbedModel = Settings.embedModel;
  // GeminiEmbeddingはデフォルトでgemini-proモデルを使用
  const geminiEmbed = new GeminiEmbedding();
  
  // グローバル設定に設定
  Settings.embedModel = geminiEmbed;
  
  // Create storage context
  const storageContext = await storageContextFromDefaults({
    persistDir: path.join(DOCS_PATH, '.indices', collectionId),
  });
  
  // Create index
  const index = await VectorStoreIndex.fromDocuments(documents, {
    storageContext,
  });
  
  // 元の設定に戻す
  Settings.embedModel = originalEmbedModel;
  
  // Save index for future use
  indices[collectionId] = { 
    index, 
    description: collection.description 
  };
  
  return index;
}

/**
 * Clone a git repository to the docs directory
 */
async function cloneRepository(repoUrl: string): Promise<string> {
  const repoName = normalizeRepoName(repoUrl);
  const repoPath = path.join(DOCS_PATH, repoName);
  
  // Check if repository already exists
  if (fs.existsSync(repoPath)) {
    // Pull latest changes
    await execAsync(`cd "${repoPath}" && git pull`);
  } else {
    // Clone repository
    await execAsync(`cd "${DOCS_PATH}" && git clone ${repoUrl}`);
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
 * Handler for listing available document collections as resources
 */
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  const collections = await listDocumentCollections();
  
  return {
    resources: collections.map(collection => ({
      uri: `docs:///${collection.id}`,
      mimeType: "text/plain",
      name: collection.name,
      description: collection.description
    }))
  };
});

/**
 * Handler for reading a document collection
 */
server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const url = new URL(request.params.uri);
  const collectionId = url.pathname.replace(/^\//, '');
  
  const collections = await listDocumentCollections();
  const collection = collections.find(c => c.id === collectionId);
  
  if (!collection) {
    throw new Error(`Document collection not found: ${collectionId}`);
  }
  
  let content: string;
  
  if (fs.statSync(collection.path).isDirectory()) {
    // List files in directory
    const files = fs.readdirSync(collection.path, { withFileTypes: true })
      .filter(entry => entry.isFile())
      .map(entry => entry.name);
    
    content = `Repository: ${collection.name}\n\nFiles:\n${files.join('\n')}`;
  } else {
    // Read file content
    content = fs.readFileSync(collection.path, 'utf-8');
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
        description: "Query a document collection using RAG",
        inputSchema: {
          type: "object",
          properties: {
            collection_id: {
              type: "string",
              description: "ID of the document collection to query"
            },
            query: {
              type: "string",
              description: "Query to run against the document collection"
            }
          },
          required: ["collection_id", "query"]
        }
      },
      {
        name: "add_git_repository",
        description: "Add a git repository to the docs directory",
        inputSchema: {
          type: "object",
          properties: {
            repository_url: {
              type: "string",
              description: "URL of the git repository to clone"
            }
          },
          required: ["repository_url"]
        }
      },
      {
        name: "add_text_file",
        description: "Add a text file to the docs directory with a specified name",
        inputSchema: {
          type: "object",
          properties: {
            file_url: {
              type: "string",
              description: "URL of the text file to download"
            },
            document_name: {
              type: "string",
              description: "Name of the document (will be used as directory name)"
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
      const collections = await listDocumentCollections();
      
      // ドキュメントの情報を整形
      const documentsList = collections.map(collection => {
        return `- ${collection.name}: ${collection.description}`;
      }).join('\n');
      
      return {
        content: [{
          type: "text",
          text: `Available documents in ${DOCS_PATH}:\n\n${documentsList}\n\nTotal documents: ${collections.length}`
        }]
      };
    }
    
    case "rag_query": {
      const collectionId = String(request.params.arguments?.collection_id);
      const query = String(request.params.arguments?.query);
      
      if (!collectionId || !query) {
        throw new Error("Collection ID and query are required");
      }
      
      // Load and index document collection if needed
      const index = await loadDocumentCollection(collectionId);
      
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
    }
    
    case "add_git_repository": {
      const repositoryUrl = String(request.params.arguments?.repository_url);
      
      if (!repositoryUrl) {
        throw new Error("Repository URL is required");
      }
      
      const repoName = await cloneRepository(repositoryUrl);
      
      return {
        content: [{
          type: "text",
          text: `Added git repository: ${repoName}`
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
          text: `Added document '${docName}' with content from ${fileUrl}`
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
        description: "Guide on how to use document collections and RAG functionality",
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

  const collections = await listDocumentCollections();
  
  return {
    messages: [
      {
        role: "user",
        content: {
          type: "text",
          text: "Please list the available document collections and guide on how to use the RAG functionality."
        }
      },
      {
        role: "assistant",
        content: {
          type: "text",
          text: `Available document collections:\n${collections.map(c => `- ${c.name}: ${c.description}`).join('\n')}\n\nUse the 'rag_query' tool to ask questions about these documents.`
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
