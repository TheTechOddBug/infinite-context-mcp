#!/usr/bin/env python3
"""
REST API Server for ChatGPT Integration
Exposes MCP functionality as HTTP endpoints for ChatGPT function calling
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import uvicorn
from main import InfiniteContextMCP

app = FastAPI(title="Infinite Context API", version="1.0.0")

# Enable CORS for ChatGPT
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MCP instance
mcp_instance = None

def get_mcp():
    """Get or create MCP instance"""
    global mcp_instance
    if mcp_instance is None:
        mcp_instance = InfiniteContextMCP()
    return mcp_instance

# Request/Response Models
class SaveContextRequest(BaseModel):
    summary: str
    topics: Optional[List[str]] = []
    key_findings: Optional[List[str]] = []
    data: Optional[Dict[str, Any]] = {}

class SearchContextRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class EnhancedSearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    use_rewrites: Optional[bool] = True
    min_relevance: Optional[float] = 0.7

class ClassifyQueryRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class RewriteQueryRequest(BaseModel):
    query: str
    rewrite_types: Optional[List[str]] = ["synonym", "broader", "expansion"]

class SmartActionRequest(BaseModel):
    request: str
    conversation_context: Optional[str] = None

class AutoCompressRequest(BaseModel):
    conversation: str
    focus: Optional[str] = "general"

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "Infinite Context API",
        "version": "1.0.0",
        "description": "REST API for ChatGPT integration",
        "endpoints": {
            "POST /save_context": "Save conversation context",
            "POST /search_context": "Search past conversations",
            "POST /enhanced_search": "Enhanced search with QU features",
            "POST /classify_query": "Classify query intent",
            "POST /rewrite_query": "Generate query rewrites",
            "POST /smart_action": "Intelligent orchestration (recommended)",
            "POST /auto_compress": "Compress conversation",
            "GET /memory_stats": "Get memory statistics",
            "GET /tools": "Get available tools (for ChatGPT function calling)"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/save_context")
async def save_context(request: SaveContextRequest):
    """Save conversation context"""
    try:
        mcp = get_mcp()
        result = await mcp.save_context({
            "summary": request.summary,
            "topics": request.topics,
            "key_findings": request.key_findings,
            "data": request.data
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search_context")
async def search_context(request: SearchContextRequest):
    """Search past conversations"""
    try:
        mcp = get_mcp()
        result = await mcp.search_context({
            "query": request.query,
            "top_k": request.top_k
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enhanced_search")
async def enhanced_search(request: EnhancedSearchRequest):
    """Enhanced search with query understanding"""
    try:
        mcp = get_mcp()
        result = await mcp.enhanced_search({
            "query": request.query,
            "top_k": request.top_k,
            "use_rewrites": request.use_rewrites,
            "min_relevance": request.min_relevance
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify_query")
async def classify_query(request: ClassifyQueryRequest):
    """Classify query intent"""
    try:
        mcp = get_mcp()
        result = await mcp.classify_query({
            "query": request.query,
            "context": request.context
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rewrite_query")
async def rewrite_query(request: RewriteQueryRequest):
    """Generate query rewrites"""
    try:
        mcp = get_mcp()
        result = await mcp.rewrite_query({
            "query": request.query,
            "rewrite_types": request.rewrite_types
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/smart_action")
async def smart_action(request: SmartActionRequest):
    """Intelligent orchestration - recommended endpoint"""
    try:
        mcp = get_mcp()
        result = await mcp.smart_action({
            "request": request.request,
            "conversation_context": request.conversation_context
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/auto_compress")
async def auto_compress(request: AutoCompressRequest):
    """Compress conversation"""
    try:
        mcp = get_mcp()
        result = await mcp.auto_compress({
            "conversation": request.conversation,
            "focus": request.focus
        })
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory_stats")
async def memory_stats():
    """Get memory statistics"""
    try:
        mcp = get_mcp()
        result = await mcp.get_memory_stats()
        return {"success": True, "result": result.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
async def get_tools():
    """Get available tools in OpenAI function calling format"""
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "save_context",
                    "description": "Save conversation context to Pinecone for later retrieval",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of the conversation"},
                            "topics": {"type": "array", "items": {"type": "string"}, "description": "Relevant topics"},
                            "key_findings": {"type": "array", "items": {"type": "string"}, "description": "Key findings or insights"},
                            "data": {"type": "object", "description": "Structured data to save"}
                        },
                        "required": ["summary"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "smart_action",
                    "description": "Intelligent orchestration tool - just describe what you want in natural language. Automatically routes to the right tools and extracts parameters.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request": {"type": "string", "description": "Natural language request (e.g., 'save this conversation about X', 'find past discussions about Y', 'show me memory stats')"},
                            "conversation_context": {"type": "string", "description": "Optional: Current conversation context to help understand the request"}
                        },
                        "required": ["request"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "enhanced_search",
                    "description": "Enhanced search with query understanding, rewrites, and guardrails. Best for finding past conversations.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "top_k": {"type": "integer", "description": "Number of results to return", "default": 3},
                            "use_rewrites": {"type": "boolean", "description": "Use query rewrites for better recall", "default": True},
                            "min_relevance": {"type": "number", "description": "Minimum relevance score threshold", "default": 0.7}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_memory_stats",
                    "description": "Get statistics about stored conversation memory",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    }

if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    print("🚀 Starting Infinite Context API Server...")
    print(f"📡 API will be available at http://0.0.0.0:{port}")
    print(f"📚 API docs at http://0.0.0.0:{port}/docs")
    print(f"🔧 Tools endpoint at http://0.0.0.0:{port}/tools")
    uvicorn.run(app, host="0.0.0.0", port=port)

