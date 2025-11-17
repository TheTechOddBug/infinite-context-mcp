# Quick Start Guide - Running the MCP Server Locally

## Prerequisites Check

✅ Virtual environment exists  
✅ Dependencies installed  
✅ .env file configured  

## Method 1: Test Run (Direct Execution)

Test the server directly to make sure it works:

```bash
cd /Users/kayajones/projects/claudecontext
source venv/bin/activate
python main.py
```

The server will start and wait for MCP client connections via stdio. Press `Ctrl+C` to stop.

## Method 2: Using the Run Script

```bash
cd /Users/kayajones/projects/claudecontext
./run_mcp.sh
```

## Method 3: Connect to Cursor/Claude Desktop

### For Cursor:

1. Open Cursor Settings
2. Go to MCP Settings (or search for "MCP")
3. Add your MCP server configuration:

```json
{
  "mcpServers": {
    "infinite-context": {
      "command": "/Users/kayajones/projects/claudecontext/run_mcp.sh",
      "args": [],
      "env": {
        "PYTHONPATH": "/Users/kayajones/projects/claudecontext"
      }
    }
  }
}
```

4. Restart Cursor
5. The MCP tools will be available in your conversations

### For Claude Desktop:

1. Open Claude Desktop Settings
2. Navigate to MCP Servers
3. Add the same configuration as above
4. Restart Claude Desktop

## Verifying It Works

Once connected, you can test the tools:

1. **Check memory stats:**
   ```
   Use the get_memory_stats tool
   ```

2. **Save context:**
   ```
   Use save_context with summary: "Test context", topics: ["test"]
   ```

3. **Search context:**
   ```
   Use search_context with query: "test"
   ```

## Troubleshooting

### Server won't start
- Check that `.env` file has all required API keys
- Verify virtual environment is activated
- Check Python version: `python --version` (should be 3.8+)

### Tools not appearing
- Restart Cursor/Claude Desktop after configuration
- Check MCP server logs for errors
- Verify the path in `mcp_config.json` is correct

### API Errors
- Verify API keys in `.env` are valid
- Check Pinecone index exists (will be created automatically)
- Ensure OpenAI API key has embedding access

## Next Steps

- Read `USAGE.md` for detailed context capture instructions
- Try the `capture_context.py` script for manual context saving
- Start saving important conversation context!




