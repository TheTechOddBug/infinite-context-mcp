# ChatGPT Integration Guide

This guide shows you how to connect the Infinite Context MCP server with ChatGPT using a REST API wrapper.

## Overview

Since MCP (Model Context Protocol) is designed for Claude Desktop/Cursor, we've created a REST API wrapper that exposes all MCP functionality as HTTP endpoints. This allows ChatGPT to use the context management features via:

1. **Function Calling** - Use OpenAI's function calling API
2. **Custom GPT Actions** - Create a custom GPT with API actions
3. **Direct API Calls** - Make HTTP requests directly

## Setup

### 1. Install Additional Dependencies

```bash
cd /Users/kayajones/projects/claudecontext
source venv/bin/activate
pip install fastapi uvicorn
```

### 2. Start the API Server

```bash
python api_server.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs (Swagger UI)
- **Tools**: http://localhost:8000/tools (OpenAI function format)

### 3. Keep Server Running

For production use, run it in the background or use a process manager:

```bash
# Background
nohup python api_server.py > api.log 2>&1 &

# Or with screen/tmux
screen -S api_server
python api_server.py
# Press Ctrl+A then D to detach
```

## Method 1: Using OpenAI Function Calling

### Step 1: Get Available Functions

```bash
curl http://localhost:8000/tools
```

This returns functions in OpenAI's format.

### Step 2: Use with ChatGPT API

```python
import openai
import requests

# Get available functions
tools_response = requests.get("http://localhost:8000/tools")
functions = tools_response.json()["tools"]

# Use with ChatGPT
client = openai.OpenAI(api_key="your-openai-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "Save this conversation about testing ChatGPT integration"}
    ],
    tools=functions,
    tool_choice="auto"
)

# Handle function calls
if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Call the API endpoint
        api_response = requests.post(
            f"http://localhost:8000/{function_name}",
            json=function_args
        )
        
        result = api_response.json()
        print(result["result"])
```

## Method 2: Custom GPT with Actions

### Step 1: Create a Custom GPT

1. Go to https://chat.openai.com/gpts
2. Click "Create" → "Create a GPT"
3. Configure your GPT

### Step 2: Add API Action

1. In the GPT editor, go to "Actions" → "Create new action"
2. Use this OpenAPI schema:

```yaml
openapi: 3.0.0
info:
  title: Infinite Context API
  version: 1.0.0
servers:
  - url: http://localhost:8000
paths:
  /smart_action:
    post:
      summary: Intelligent orchestration - describe what you want
      operationId: smartAction
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                request:
                  type: string
                  description: Natural language request
                conversation_context:
                  type: string
                  description: Optional conversation context
              required:
                - request
      responses:
        '200':
          description: Success
          content:
            application/json:
              schema:
                type: object
                properties:
                  success:
                    type: boolean
                  result:
                    type: string
  /enhanced_search:
    post:
      summary: Enhanced search with query understanding
      operationId: enhancedSearch
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                top_k:
                  type: integer
                  default: 3
      responses:
        '200':
          description: Success
  /save_context:
    post:
      summary: Save conversation context
      operationId: saveContext
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                summary:
                  type: string
                topics:
                  type: array
                  items:
                    type: string
      responses:
        '200':
          description: Success
  /memory_stats:
    get:
      summary: Get memory statistics
      operationId: memoryStats
      responses:
        '200':
          description: Success
```

### Step 3: Test Your Custom GPT

```
User: "Save this conversation about integrating with ChatGPT"
GPT: [Calls smart_action API] ✅ Context saved!
```

## Method 3: Direct API Usage

### Example: Save Context

```python
import requests

response = requests.post(
    "http://localhost:8000/smart_action",
    json={
        "request": "save this conversation about ChatGPT integration",
        "conversation_context": "We discussed creating a REST API wrapper..."
    }
)

print(response.json()["result"])
```

### Example: Search Context

```python
response = requests.post(
    "http://localhost:8000/enhanced_search",
    json={
        "query": "ChatGPT integration",
        "top_k": 5
    }
)

print(response.json()["result"])
```

### Example: Get Stats

```python
response = requests.get("http://localhost:8000/memory_stats")
print(response.json()["result"])
```

## Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/smart_action` | POST | **Recommended** - Intelligent orchestration |
| `/save_context` | POST | Save conversation context |
| `/enhanced_search` | POST | Enhanced search with QU features |
| `/search_context` | POST | Basic search |
| `/classify_query` | POST | Classify query intent |
| `/rewrite_query` | POST | Generate query rewrites |
| `/auto_compress` | POST | Compress conversation |
| `/memory_stats` | GET | Get memory statistics |
| `/tools` | GET | Get OpenAI function definitions |
| `/docs` | GET | Swagger API documentation |

## Using with ngrok (for Remote Access)

If you want to use the API from ChatGPT's cloud (not just localhost):

### 1. Install ngrok

```bash
brew install ngrok  # macOS
# or download from https://ngrok.com/
```

### 2. Start ngrok tunnel

```bash
ngrok http 8000
```

### 3. Use the ngrok URL

Use the HTTPS URL (e.g., `https://abc123.ngrok.io`) in your Custom GPT actions instead of `localhost:8000`.

## Example: Complete ChatGPT Integration Script

```python
#!/usr/bin/env python3
"""
Example: Using Infinite Context API with ChatGPT
"""
import openai
import requests
import json

API_BASE = "http://localhost:8000"

def get_tools():
    """Get available tools"""
    response = requests.get(f"{API_BASE}/tools")
    return response.json()["tools"]

def call_api_function(function_name, arguments):
    """Call API function"""
    response = requests.post(
        f"{API_BASE}/{function_name}",
        json=arguments
    )
    return response.json()

# Initialize ChatGPT client
client = openai.OpenAI(api_key="your-api-key")

# Get available functions
functions = get_tools()

# Chat with ChatGPT
messages = [
    {"role": "user", "content": "Save this conversation about ChatGPT integration"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=functions,
    tool_choice="auto"
)

# Handle function calls
message = response.choices[0].message
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
        
        # Call our API
        result = call_api_function(function_name, function_args)
        
        # Add result to conversation
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result["result"]
        })
        
        # Get ChatGPT's response
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        print(response.choices[0].message.content)
```

## Testing the API

### Using curl

```bash
# Test smart_action
curl -X POST http://localhost:8000/smart_action \
  -H "Content-Type: application/json" \
  -d '{"request": "save this test conversation"}'

# Test enhanced_search
curl -X POST http://localhost:8000/enhanced_search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 3}'

# Get stats
curl http://localhost:8000/memory_stats
```

### Using Python requests

```python
import requests

# Smart action (recommended)
response = requests.post(
    "http://localhost:8000/smart_action",
    json={"request": "find past discussions about MCP"}
)
print(response.json()["result"])
```

## Troubleshooting

### API Server Won't Start

- Check if port 8000 is available: `lsof -i :8000`
- Make sure dependencies are installed: `pip install fastapi uvicorn`
- Check API logs for errors

### ChatGPT Can't Connect

- Make sure API server is running
- If using Custom GPT, use ngrok for remote access
- Check firewall settings
- Verify API endpoint URLs

### Function Calls Not Working

- Verify `/tools` endpoint returns correct format
- Check that function names match API endpoints
- Ensure request body matches expected schema

## Security Notes

⚠️ **Important**: The API server runs on `0.0.0.0` by default, making it accessible from any network interface. For production:

1. Use authentication (API keys, OAuth)
2. Run behind a reverse proxy (nginx)
3. Use HTTPS (with SSL certificates)
4. Restrict access with firewall rules
5. Don't expose sensitive endpoints publicly

## Next Steps

1. Start the API server: `python api_server.py`
2. Test endpoints: Visit http://localhost:8000/docs
3. Integrate with ChatGPT using one of the methods above
4. Use `smart_action` for frictionless context management!

## Support

For issues or questions:
- Check API docs at http://localhost:8000/docs
- Review [SMART_ACTION.md](SMART_ACTION.md) for usage examples
- See [README.md](README.md) for general documentation

