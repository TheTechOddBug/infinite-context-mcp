# Infinite Context MCP

A unified context layer that makes long AI threads usable across tools. It stores key facts and conversation history in Pinecone, auto-compresses before token limits, and lets Claude retrieve the exact slice you need—decisions, constraints, summaries—without re-explaining threads.

## Features

- **Infinite Context**: Automatically compresses long conversations while preserving important information
- **Vector Search**: Uses Pinecone for semantic search across conversation history
- **Full Content Storage**: Stores up to ~5,000 words of raw content per chunk, preserving formatting
- **Smart Compression**: Uses LLMs to intelligently summarize conversations
- **High-Quality Embeddings**: Uses OpenAI's text-embedding-3-large model
- **Query Understanding**: Instacart-inspired query classification, rewrites, and enhanced search
- **Smart Action**: Natural language orchestration - just describe what you want

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Keys

Copy the example environment file and fill in your API keys:

```bash
cp env_example.txt .env
```

Edit `.env` and add your actual API keys:

```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_custom_index_name_here  # Optional, defaults to "infinite-context-index"
```

### 3. Get API Keys

- **Anthropic**: Get your API key from [Anthropic Console](https://console.anthropic.com/)
- **OpenAI**: Get your API key from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Pinecone**: Get your API key from [Pinecone Console](https://app.pinecone.io/)

### 4. Add MCP Server to Claude Code

Add the MCP server to Claude Code using the command line:

```bash
claude mcp add --transport stdio infinite-context -- /Users/kayajones/projects/claudecontext/run_mcp.sh
```

**Note**: Update the path to match your project location.

### 5. Restart Claude Code

Completely quit and restart Claude Code to load the MCP server.

## Usage

Once the MCP server is connected, you can use the tools directly in your Claude conversations.

### Using Smart Action (Recommended)

The easiest way to use the MCP - just describe what you want in natural language:

```
You: "Use smart_action to save this conversation about setting up the MCP server"

Claude: [Automatically calls smart_action tool]
        ✅ Context saved with topics: MCP, Setup, Configuration
```

### Quick Reference

| What You Want | How to Ask Claude |
|--------------|-------------------|
| Save context | `"Use smart_action to save this conversation about X"` |
| Find context | `"Use smart_action to find past discussions about Y"` |
| Get stats | `"Use get_memory_stats to show memory statistics"` |
| Search | `"Use enhanced_search with query: 'your search term'"` |

### Available MCP Tools

#### Core Tools

- **`smart_action`** ⭐ - Intelligent orchestration tool. Just describe what you want in natural language.
- **`save_context`** - Save conversation context to Pinecone (includes full content up to ~5,000 words)
- **`search_context`** - Basic semantic search across saved conversations
- **`enhanced_search`** - Advanced search with query understanding, rewrites, and guardrails
- **`get_memory_stats`** - View memory statistics and cache performance

#### Query Understanding Tools

- **`classify_query`** - Classify queries to understand intent and categories
- **`rewrite_query`** - Generate query rewrites (synonyms, broader terms, expansions)

#### Advanced Tools

- **`auto_compress`** - Automatically compress and save long conversations

### Example Usage in Claude

**Save Context:**
```
You: "Use smart_action to save this conversation about ERP integration requirements"

Claude: [Calls smart_action automatically]
        ✅ Saved context chunk 20250101_120000_chunk_0
        Topics: ERP, Integration, Requirements
```

**Search Context:**
```
You: "Use enhanced_search to find information about WMS sync requirements"

Claude: [Calls enhanced_search automatically]
        🚀 Enhanced Search Results:
        📄 Result 1 (relevance: 85%)
           📝 Summary: Case Study: ERP -> WMS Integration...
           📜 Content Preview: [Full formatted content shown]
```

**Get Statistics:**
```
You: "Use get_memory_stats to show me memory usage"

Claude: [Calls get_memory_stats automatically]
        📊 Memory Statistics:
        Total vectors stored: 42
        Current session: 20250101_120000
        ...
```

## How It Works

1. **Context Storage**: When you save context, the system stores:
   - Summary and topics (for quick reference)
   - Full raw content (up to ~5,000 words, preserving formatting)
   - Key findings and structured data
   - All stored as vectors in Pinecone for semantic search

2. **Semantic Search**: Uses OpenAI embeddings to find relevant context even with different wording or formats

3. **Query Understanding**: Automatically classifies queries, generates rewrites, and applies guardrails for better results

4. **Smart Action**: Intelligently routes your natural language requests to the right tools

## Storage Format

Each saved context chunk includes:

- **Vector Embedding**: 1024-dimensional embedding for semantic search
- **Metadata**:
  - `summary`: Brief summary (up to 2,000 chars)
  - `content`: Full raw content (up to 25,000 chars) - preserves all formatting
  - `topics`: List of topics/tags
  - `findings`: Key findings or important points
  - `data`: Structured data (JSON, up to 2,000 chars)
  - `timestamp`: ISO format timestamp
  - `session_id`: Session identifier

## Troubleshooting

### MCP Server Not Appearing

- **Check the script path**: Make sure the path in the `claude mcp add` command is correct
- **Check Python environment**: Ensure the virtual environment has all dependencies installed
- **Check .env file**: Verify all API keys are set correctly
- **Restart Claude Code**: Fully quit and restart (not just reload)

### Tools Not Available

- **Restart Claude Code**: Fully quit and restart after adding the MCP server
- **Check MCP connection**: Verify the server is listed as connected in Claude Code settings
- **Test script manually**: Run `./run_mcp.sh` directly to ensure it works

### Permission Errors

```bash
chmod +x /Users/kayajones/projects/claudecontext/run_mcp.sh
```

### API Errors

- **Verify API keys**: Check that all three API keys in `.env` are valid
- **Check Pinecone index**: The system will automatically create the Pinecone index on first run
- **Ensure OpenAI API key has embedding access**: Required for generating embeddings

## Security

**Important Security Notes:**

- Never commit your `.env` file to version control
- The `.gitignore` file is configured to exclude sensitive files
- All API keys are loaded from environment variables
- The Pinecone index name is configurable via environment variable
- Use unique index names for different projects to avoid data conflicts

## Additional Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick start guide for running the MCP server
- [SMART_ACTION.md](SMART_ACTION.md) - Detailed guide to smart_action orchestration
- [QUERY_UNDERSTANDING.md](QUERY_UNDERSTANDING.md) - Guide to Query Understanding features
- [USAGE.md](USAGE.md) - How to capture and use context effectively
