# Unified Context Layer MCP

A  Memory System for AI assistants that goes beyond traditional RAG. Store conversations, build knowledge graphs, extract facts, and maintain user profiles—all searchable with semantic understanding.Use across your apps and LLMs. 


##  Features

### Core Memory
- **Infinite Context** - Automatically compress long conversations while preserving important information
- **Vector Search** - Semantic search across all saved conversations using Pinecone
- **Full Content Storage** - Store up to ~5,000 words per chunk with formatting preserved
- **Smart Compression** - LLM-powered intelligent summarization

### Memory System 
- **User Profiles** - Automatically learns your interests, projects, and preferences
- **Entity Graphs** - Build knowledge graphs connecting people, projects, concepts
- **Fact Extraction** - Extract and chain atomic facts from conversations
- **Temporal Awareness** - Understand recency and relevance over time
- **Hybrid Scoring** - Combine semantic similarity with temporal and entity signals

### Query Understanding
- **Query Classification** - Understand intent (search, save, question, etc.)
- **Query Rewrites** - Generate synonyms, broader terms, and expansions
- **Enhanced Search** - Guardrails, auto-refinement, and follow-up recommendations

### Indexing
- **GitHub Repositories** - Index entire repos for code search
- **Documentation Sites** - Crawl and index docs
- **Websites** - Full website crawling
- **Local Filesystems** - Index local directories
- **Single URLs** - Index individual pages (blogs, ChatGPT conversations, tweets)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/infinite-context-mcp.git
cd infinite-context-mcp
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

```bash
cp env_example.txt .env
```

Edit `.env` with your API keys:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=infinite-context-index  # Optional, this is the default
```

**Get your API keys:**
- [Anthropic Console](https://console.anthropic.com/) - For Claude (AI responses)
- [OpenAI Platform](https://platform.openai.com/api-keys) - For embeddings
- [Pinecone Console](https://app.pinecone.io/) - For vector storage (free tier available)

### 5. Update the Run Script

Edit `run_mcp.sh` to point to your installation:

```bash
#!/bin/bash
cd /path/to/your/infinite-context-mcp
source venv/bin/activate
exec python main.py
```

Make it executable:

```bash
chmod +x run_mcp.sh
```

### 6. Connect to Your AI Tool

#### For Cursor

Add to your Cursor MCP settings (`~/.cursor/mcp.json` or via Settings > MCP):

```json
{
  "mcpServers": {
    "infinite-context": {
      "command": "/path/to/your/infinite-context-mcp/run_mcp.sh",
      "args": [],
      "env": {
        "PYTHONPATH": "/path/to/your/infinite-context-mcp"
      }
    }
  }
}
```

#### For Claude Desktop

Add the same configuration to Claude Desktop's MCP settings.

#### For Claude Code (CLI)

```bash
claude mcp add --transport stdio infinite-context -- /path/to/your/infinite-context-mcp/run_mcp.sh
```

### 7. Restart Your AI Tool

Fully quit and restart Cursor/Claude Desktop to load the MCP server.

---

## 📖 Usage

### Smart Action (Recommended)

The easiest way to use the MCP—just describe what you want:

```
"Save this conversation about setting up the MCP server"
"Find past discussions about API integration"
"What have I been working on lately?"
"Show me my user profile"
```

### Quick Reference

| What You Want | Example Request |
|---------------|-----------------|
| Save context | `"Save this conversation about X"` |
| Search context | `"Find information about Y"` |
| Ask a question | `"What do I know about Z?"` |
| Get profile | `"Show my user profile"` |
| Memory stats | `"Show memory statistics"` |
| Index a repo | `"Index https://github.com/owner/repo"` |

---

## 🛠️ Available Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `smart_action` ⭐ | Intelligent orchestration—just describe what you want |
| `save_context` | Save conversation context with summary, topics, and full content |
| `search_context` | Semantic search across saved conversations |
| `enhanced_search` | Advanced search with query understanding and auto-refinement |
| `ask_question` | RAG Q&A—ask questions about your saved data |
| `auto_compress` | Compress and save long conversations |
| `get_memory_stats` | View memory statistics |

### Memory System Tools

| Tool | Description |
|------|-------------|
| `get_user_profile` | View your learned profile (interests, focus, stats) |
| `update_user_profile` | Manually update profile preferences and focus |
| `query_knowledge_graph` | Find entity relationships and connections |
| `get_graph_summary` | Overview of your knowledge graph |
| `query_facts` | Search extracted facts by entity or type |
| `get_fact_summary` | Summary of all extracted facts |

### Query Understanding Tools

| Tool | Description |
|------|-------------|
| `classify_query` | Classify query intent and categories |
| `rewrite_query` | Generate query variations for better recall |

### Indexing Tools

| Tool | Description |
|------|-------------|
| `index_repository` | Index a GitHub repository |
| `index_documentation` | Index a documentation site |
| `index_website` | Crawl and index a full website |
| `index_local_filesystem` | Index a local directory |
| `index_url` | Index a single URL (any type) |
| `check_indexing_status` | Check status of indexing job |
| `list_indexed_sources` | List all indexed sources |
| `delete_indexed_source` | Remove an indexed source |

---

## 🧠 How the Memory System Works

### 1. Context Storage

When you save context, the system stores:
- **Summary** - Brief overview (up to 2,000 chars)
- **Full Content** - Raw content preserving formatting (up to 25,000 chars)
- **Topics** - Tags for categorization
- **Key Findings** - Important points extracted
- **Entities** - People, projects, concepts mentioned
- **Facts** - Atomic facts for precise retrieval

### 2. User Profile Learning

The system automatically learns:
- **Interests** - Topics you frequently discuss
- **Current Focus** - What you're actively working on
- **Entity Connections** - How concepts in your work relate

### 3. Knowledge Graph

As you save contexts, entities are extracted and connected:
- Find relationships between projects, people, and concepts
- Discover paths between entities
- Get summaries of your knowledge domain

### 4. Fact Extraction

Atomic facts are extracted with:
- **Type** - Statement, decision, preference, problem, solution, etc.
- **Confidence** - How certain the extraction is
- **Temporal Ordering** - Newer facts can supersede older ones

### 5. Hybrid Search

Search combines multiple signals:
- **Semantic Similarity** - Meaning-based matching
- **Temporal Relevance** - Recent content weighted higher
- **Entity Matching** - Boost results with matching entities
- **Profile Context** - Personalized based on your interests

---

## 📁 Project Structure

```
infinite-context-mcp/
├── main.py                 # MCP server with all tools
├── run_mcp.sh              # Startup script
├── requirements.txt        # Python dependencies
├── env_example.txt         # Environment variable template
├── query_understanding.py  # Query classification & rewriting
├── user_profile.py         # User profile management
├── entity_graph.py         # Knowledge graph implementation
├── fact_chain.py           # Fact extraction & chaining
├── memory_scorer.py        # Hybrid scoring system
├── lib/
│   └── indexing_service.py # External indexing service
└── api/
    └── api_server.py       # Optional REST API
```

---

## 🔧 Troubleshooting

### MCP Server Not Appearing

1. **Check the script path** - Verify `run_mcp.sh` path in your MCP config
2. **Test the script** - Run `./run_mcp.sh` directly to see errors
3. **Check Python environment** - Ensure venv has all dependencies
4. **Restart completely** - Fully quit and restart your AI tool

### API Errors

1. **Verify API keys** - Check all keys in `.env` are valid
2. **Pinecone index** - Will be auto-created on first run
3. **OpenAI access** - Ensure your key has embedding API access

### Permission Errors

```bash
chmod +x run_mcp.sh
```

### Tools Not Working

1. Check MCP connection status in your AI tool's settings
2. Look for error messages in the terminal running the MCP
3. Verify the Pinecone index exists and is accessible

---

## 🔐 Security

- **Never commit `.env`** - It's in `.gitignore` by default
- **API keys in environment** - All secrets loaded from env vars
- **Unique index names** - Use different `PINECONE_INDEX_NAME` for different projects
- **Local storage** - User profiles stored locally in `~/.infinite-context/`

---

## 📚 Additional Documentation

- [QUICKSTART.md](QUICKSTART.md) - Quick setup guide
- [SMART_ACTION.md](SMART_ACTION.md) - Smart action orchestration details
- [QUERY_UNDERSTANDING.md](QUERY_UNDERSTANDING.md) - Query understanding features
- [USAGE.md](USAGE.md) - Detailed usage guide

---

## 🤝 Contributing

Contributions welcome!

---

## 📄 License

MIT License 

## Contact 
Reach out to me on x!
