# mcp-docs-rag MCP Server

RAG (Retrieval-Augmented Generation) for documents in a local directory

This is a TypeScript-based MCP server that implements a RAG system for documents stored in a local directory. It allows users to query documents using LLMs with context from locally stored repositories and text files.

## Features

### Resources
- List and access document collections via `docs://` URIs
- Documents can be Git repositories or text files
- Plain text mime type for content access

### Tools
- `rag_query` - Query document collections using RAG
  - Takes collection_id and query as parameters
  - Returns AI-generated responses with context from documents
- `add_git_repository` - Clone a Git repository to the docs directory
  - Takes repository_url as parameter
  - Automatically pulls latest changes if repository already exists
- `add_text_file` - Download a text file to the docs directory
  - Takes file_url as parameter
  - Uses wget to download file

### Prompts
- `summarize_collection` - List available document collections
  - Includes list of available document collections
  - Provides usage hints for RAG functionality

## Development

Install dependencies:
```bash
npm install
```

Build the server:
```bash
npm run build
```

For development with auto-rebuild:
```bash
npm run watch
```

## Setup

This server requires a local directory for storing documents. By default, it uses `~/docs` but you can configure a different location with the `DOCS_PATH` environment variable.

### Document Structure

The documents directory can contain:
- Git repositories (cloned directories)
- Plain text files (with .txt extension)

Each document collection is indexed separately using llama-index.ts with Google's Gemini embeddings.

### API Keys

This server uses Google's Gemini API for document indexing and querying. You need to set your Gemini API key as an environment variable:

```bash
export GEMINI_API_KEY=your-api-key-here
```

You can obtain a Gemini API key from the [Google AI Studio](https://makersuite.google.com/app/apikey) website. Add this key to your shell profile or include it in the environment configuration for Claude Desktop.

## Installation

To use with Claude Desktop, add the server config:

On MacOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
On Windows: `%APPDATA%/Claude/claude_desktop_config.json`
On Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "docs-rag": {
      "command": "npx",
      "args": ["-y", "@kazuph/mcp-docs-rag"],
      "env": {
        "DOCS_PATH": "/Users/username/docs",
        "GEMINI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

Make sure to replace `/Users/username/docs` with the actual path to your documents directory.

### Debugging

Since MCP servers communicate over stdio, debugging can be challenging. We recommend using the [MCP Inspector](https://github.com/modelcontextprotocol/inspector), which is available as a package script:

```bash
npm run inspector
```

The Inspector will provide a URL to access debugging tools in your browser.

## Usage

Once configured, you can use the server with Claude to:

1. **Add documents**:
   ```
   Add a new document from GitHub: https://github.com/username/repository
   ```
   or
   ```
   Add this text file: https://example.com/document.txt
   ```

2. **Query documents**:
   ```
   What does the documentation say about X in the Y repository?
   ```

3. **List available documents**:
   ```
   What documents do you have access to?
   ```

The server will automatically handle indexing of documents for efficient retrieval.
