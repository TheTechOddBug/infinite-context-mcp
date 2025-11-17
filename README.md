# Infinite Context Claude

A sophisticated chat system that uses Claude with infinite context through intelligent conversation compression and vector storage using Pinecone.

## Features

- **Infinite Context**: Automatically compresses long conversations while preserving important information
- **Vector Search**: Uses Pinecone for semantic search across conversation history
- **Smart Compression**: Uses Claude to intelligently summarize conversations
- **High-Quality Embeddings**: Uses OpenAI's text-embedding-3-large model
- **Query Understanding** (NEW): Instacart-inspired query classification, rewrites, and enhanced search
  - Query classification to understand intent and categories
  - Query rewrites (synonyms, broader terms, expansions) for better recall
  - Post-processing guardrails to filter low-quality results
  - Hybrid caching system for performance

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

### 4. Run the Application

```bash
python main.py
```

## Getting Started Examples

### Example 1: Save Context (Smart Action)

The easiest way to save context - just describe what you want:

```python
from main import InfiniteContextMCP
import asyncio

async def save_example():
    mcp = InfiniteContextMCP()
    
    # Smart action automatically understands intent and extracts parameters
    result = await mcp.smart_action({
        "request": "save this conversation about setting up the MCP server with Pinecone integration",
        "conversation_context": "We discussed integrating Pinecone for vector storage..."
    })
    print(result.text)

asyncio.run(save_example())
```

**Output:**
```
🎯 Smart Action: save this conversation about setting up the MCP server...
📋 Detected Action: save_context
⚙️  Parameters: {
  "summary": "save this conversation about setting up the MCP server...",
  "topics": ["MCP", "Pinecone", "Vector Storage"]
}
============================================================

✅ Saved context chunk 20250101_120000_chunk_0
Session: 20250101_120000
Topics: MCP, Pinecone, Vector Storage
Findings: 0 key findings stored
```

### Example 2: Search Context (Enhanced Search)

Find past conversations with intelligent query understanding:

```python
async def search_example():
    mcp = InfiniteContextMCP()
    
    # Smart action uses enhanced_search automatically
    result = await mcp.smart_action({
        "request": "find past discussions about query understanding"
    })
    print(result.text)

asyncio.run(search_example())
```

**Or use enhanced_search directly:**

```python
async def enhanced_search_example():
    mcp = InfiniteContextMCP()
    
    result = await mcp.enhanced_search({
        "query": "MCP server configuration",
        "top_k": 5,
        "use_rewrites": True,
        "min_relevance": 0.7
    })
    print(result.text)

asyncio.run(enhanced_search_example())
```

### Example 3: Query Classification

Understand what type of query you have:

```python
async def classify_example():
    mcp = InfiniteContextMCP()
    
    result = await mcp.classify_query({
        "query": "how do I save context to Pinecone?"
    })
    print(result.text)

asyncio.run(classify_example())
```

**Output:**
```
🎯 Query Classification for: 'how do I save context to Pinecone?'

Query Type: question
Confidence: 95.0%
Intent: User wants to know how to save context data to Pinecone
Categories: Pinecone, Context Management, Storage
```

### Example 4: Query Rewrites

Generate alternative queries for better recall:

```python
async def rewrite_example():
    mcp = InfiniteContextMCP()
    
    result = await mcp.rewrite_query({
        "query": "MCP configuration",
        "rewrite_types": ["synonym", "broader", "expansion"]
    })
    print(result.text)

asyncio.run(rewrite_example())
```

**Output:**
```
✏️ Query Rewrites for: 'MCP configuration'

Generated 3 rewrites:

1. [SYNONYM] MCP setup
   Confidence: 90.0%
   Reasoning: Alternative phrasing with same meaning

2. [BROADER] MCP server setup
   Confidence: 85.0%
   Reasoning: More general query that encompasses the original

3. [EXPANSION] MCP server configuration and setup
   Confidence: 80.0%
   Reasoning: Expanded with related terms to improve recall
```

### Example 5: Multi-Action Workflow

Chain multiple actions together:

```python
async def multi_action_example():
    mcp = InfiniteContextMCP()
    
    # Search and save in one request
    result = await mcp.smart_action({
        "request": "search for context about Pinecone and save the top 3 results"
    })
    print(result.text)

asyncio.run(multi_action_example())
```

### Example 6: Memory Statistics

Check your memory usage:

```python
async def stats_example():
    mcp = InfiniteContextMCP()
    
    result = await mcp.get_memory_stats()
    print(result.text)

asyncio.run(stats_example())
```

**Output:**
```
📊 Memory Statistics:
Total vectors stored: 42
Current session: 20250101_120000
Chunks in session: 5
Index dimension: 1024
Index fullness: 0.05%

🔍 Query Understanding Cache:
   Cache size: 15
   Cache hits: 23
   Cache misses: 8
   Hit rate: 74.2%
```

### Example 7: Using MCP Tools in Claude/Cursor

When the MCP server is running, you can use tools directly in your conversation:

```
User: "Use smart_action to save this conversation about integrating 
       Instacart's query understanding system"

Claude: [Calls smart_action tool automatically]
        ✅ Context saved with topics: Query Understanding, Instacart, RAG
```

### Quick Reference

| What You Want | Smart Action Request |
|--------------|---------------------|
| Save context | `"save this conversation about X"` |
| Find context | `"find past discussions about Y"` |
| Get stats | `"show me memory statistics"` |
| Classify query | `"what type of query is Z?"` |
| Generate rewrites | `"give me alternatives for query Q"` |
| Multi-action | `"search for X and save the top 3"` |

## How It Works

1. **Context Management**: Monitors token usage and compresses conversations when they approach Claude's limit
2. **Intelligent Compression**: Uses Claude to extract key information including:
   - Founder data and company associations
   - Duplicate detection across systems
   - Decisions and actions taken
   - Statistics and metrics
3. **Vector Storage**: Stores compressed conversations in Pinecone for semantic search
4. **Context Retrieval**: Searches past conversations to provide relevant context for new questions

## Usage

### Basic MCP Tools

The MCP server provides several tools for context management:

- `save_context`: Save conversation context to Pinecone
- `search_context`: Search past conversations
- `auto_compress`: Automatically compress long conversations
- `get_memory_stats`: View memory statistics

### Query Understanding Tools (NEW)

Inspired by [Instacart's Intent Engine](https://tech.instacart.com/building-the-intent-engine-how-instacart-is-revamping-query-understanding-with-llms-3ac8051ae7ac):

- `classify_query`: Classify queries to understand intent and categories
- `rewrite_query`: Generate query rewrites (synonyms, broader terms, expansions)
- `enhanced_search`: Comprehensive search with query understanding, rewrites, and guardrails

### Smart Action Orchestration (NEW) ⭐

**Frictionless Context Management**: The `smart_action` tool intelligently orchestrates all tools automatically. Just describe what you want in natural language!

```python
# Instead of calling multiple tools manually:
smart_action({
    "request": "find and save context about MCP setup"
})
```

The system automatically:
- Understands your intent
- Routes to the right tools
- Extracts parameters intelligently
- Orchestrates multiple actions when needed

See [SMART_ACTION.md](SMART_ACTION.md) for detailed documentation.

### Testing

Run the test suite to see Query Understanding in action:

```bash
python test_query_understanding.py
```

## Configuration

Key parameters you can adjust in the code:

- `MAX_TOKENS`: Claude's context limit (default: 150,000)
- `THRESHOLD`: Compression trigger percentage (default: 75%)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: "infinite-context-index")
- `embedding_model`: OpenAI embedding model (default: "text-embedding-3-large")

## Security

**Important Security Notes:**

- Never commit your `.env` file to version control
- The `.gitignore` file is configured to exclude sensitive files
- All API keys are loaded from environment variables
- The Pinecone index name is configurable via environment variable
- Use unique index names for different projects to avoid data conflicts

## Troubleshooting

- **Missing API Keys**: Make sure all three API keys are set in your `.env` file
- **Pinecone Index**: The system will automatically create the Pinecone index on first run
- **Rate Limits**: The system includes caching to minimize API calls
- **Query Understanding**: See [QUERY_UNDERSTANDING.md](QUERY_UNDERSTANDING.md) for details on the new QU features

## Additional Documentation

- [SMART_ACTION.md](SMART_ACTION.md) - **NEW**: Intelligent orchestration for frictionless context management
- [QUERY_UNDERSTANDING.md](QUERY_UNDERSTANDING.md) - Detailed guide to Query Understanding features
- [USAGE.md](USAGE.md) - How to capture and use context
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
