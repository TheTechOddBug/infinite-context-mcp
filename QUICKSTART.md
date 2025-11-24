# Quick Start Guide

Get the Infinite Context MCP running in 5 minutes.

## Prerequisites

- Python 3.8+
- API keys for Anthropic, OpenAI, and Pinecone

## Setup

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/infinite-context-mcp.git
cd infinite-context-mcp

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp env_example.txt .env
```

Edit `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
```

### 3. Update Run Script

Edit `run_mcp.sh` with your actual path:

```bash
#!/bin/bash
cd /path/to/your/infinite-context-mcp
source venv/bin/activate
exec python main.py
```

Make executable:

```bash
chmod +x run_mcp.sh
```

### 4. Test It Works

```bash
./run_mcp.sh
```

The server will start and wait for MCP client connections. Press `Ctrl+C` to stop.

## Connect to Your AI Tool

### Cursor

Add to `~/.cursor/mcp.json` (or Settings > MCP):

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

Restart Cursor completely.

### Claude Desktop

Same configuration in Claude Desktop's MCP settings.

### Claude Code (CLI)

```bash
claude mcp add --transport stdio infinite-context -- /path/to/your/infinite-context-mcp/run_mcp.sh
```

## Verify It Works

Once connected, try these in your AI conversation:

1. **Check memory stats:**
   ```
   Show me memory statistics
   ```

2. **Save context:**
   ```
   Save this conversation about getting started with the MCP
   ```

3. **Search context:**
   ```
   Find information about MCP setup
   ```

4. **View your profile:**
   ```
   Show my user profile
   ```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Server won't start | Check `.env` has all API keys |
| Tools not appearing | Restart Cursor/Claude completely |
| API errors | Verify API keys are valid |
| Permission denied | Run `chmod +x run_mcp.sh` |

## Next Steps

- Read the full [README.md](README.md) for all features
- Try the Memory System tools (profiles, knowledge graph, facts)
- Index a GitHub repo or documentation site
- See [USAGE.md](USAGE.md) for detailed examples
