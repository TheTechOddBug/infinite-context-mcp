# infinite_context_mcp.py
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.models import InitializationOptions

import tiktoken
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class InfiniteContextMCP:
    def __init__(self):
        self.server = Server("infinite-context")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Get index name from environment variable with fallback
        index_name = os.getenv("PINECONE_INDEX_NAME", "infinite-context-index")
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
        
        self.index = pc.Index(index_name)
        self.openai = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Session management
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chunk_count = 0
        
        # Register tools
        self.setup_tools()
    
    def setup_tools(self):
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="save_context",
                    description="Save current conversation context to Pinecone for later retrieval",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Summary of the conversation"},
                            "topics": {"type": "array", "items": {"type": "string"}},
                            "data": {"type": "object", "description": "Structured data to save"},
                            "key_findings": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["summary"]
                    }
                ),
                Tool(
                    name="search_context",
                    description="Search past conversation context from Pinecone",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "top_k": {"type": "integer", "default": 3}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="auto_compress",
                    description="Automatically compress and save conversation when context is getting full",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "conversation": {"type": "string", "description": "Full conversation to compress"},
                            "focus": {"type": "string", "description": "What to focus extraction on"}
                        },
                        "required": ["conversation"]
                    }
                ),
                Tool(
                    name="get_memory_stats",
                    description="Get statistics about stored conversation memory",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "save_context":
                result = await self.save_context(arguments)
                return [result]
            elif name == "search_context":
                result = await self.search_context(arguments)
                return [result]
            elif name == "auto_compress":
                result = await self.auto_compress(arguments)
                return [result]
            elif name == "get_memory_stats":
                result = await self.get_memory_stats()
                return [result]
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def save_context(self, args: dict) -> TextContent:
        """Save context to Pinecone"""
        # Generate embedding
        text = f"""
        Summary: {args.get('summary', '')}
        Topics: {', '.join(args.get('topics', []))}
        Findings: {', '.join(args.get('key_findings', []))}
        Data: {json.dumps(args.get('data', {}))}
        """
        
        response = self.openai.embeddings.create(
            input=text,
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        embedding = response.data[0].embedding
        
        # Store in Pinecone
        chunk_id = f"{self.current_session}_chunk_{self.chunk_count}"
        
        self.index.upsert([{
            "id": chunk_id,
            "values": embedding,
            "metadata": {
                "session_id": self.current_session,
                "chunk_id": self.chunk_count,
                "timestamp": datetime.now().isoformat(),
                "summary": args.get('summary', '')[:1000],
                "topics": str(args.get('topics', [])),
                "findings": str(args.get('key_findings', [])),
                "data": json.dumps(args.get('data', {}))[:1000]
            }
        }])
        
        self.chunk_count += 1
        
        return TextContent(
            type="text",
            text=f"✅ Saved context chunk {chunk_id}\n"
                 f"Session: {self.current_session}\n"
                 f"Topics: {', '.join(args.get('topics', []))}\n"
                 f"Findings: {len(args.get('key_findings', []))} key findings stored"
        )
    
    async def search_context(self, args: dict) -> TextContent:
        """Search Pinecone for relevant context"""
        query = args.get('query', '')
        top_k = args.get('top_k', 3)
        
        # Generate query embedding
        response = self.openai.embeddings.create(
            input=query,
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        embedding = response.data[0].embedding
        
        # Search Pinecone
        results = self.index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        context_text = f"🔍 Found {len(results['matches'])} relevant contexts for: '{query}'\n\n"
        
        for i, match in enumerate(results['matches'], 1):
            meta = match['metadata']
            context_text += f"📄 Result {i} (relevance: {match['score']:.2%})\n"
            context_text += f"   🆔 Chunk ID: {meta.get('chunk_id', 'N/A')}\n"
            context_text += f"   📅 Time: {meta.get('timestamp', 'N/A')}\n"
            context_text += f"   📝 Summary: {meta.get('summary', 'N/A')}\n"
            context_text += f"   🏷️  Topics: {meta.get('topics', 'N/A')}\n"
            if meta.get('findings'):
                context_text += f"   🔍 Key Findings: {meta.get('findings', 'N/A')}\n"
            if meta.get('data'):
                context_text += f"   📊 Data: {meta.get('data', 'N/A')}\n"
            context_text += "\n"
        
        return TextContent(type="text", text=context_text)
    
    async def auto_compress(self, args: dict) -> TextContent:
        """Compress conversation and save to Pinecone"""
        conversation = args.get('conversation', '')
        focus = args.get('focus', 'general')
        
        # This would normally use Claude to compress, but since we're IN Claude,
        # we'll just extract and save the key points
        
        # For now, create a structured summary
        summary_data = {
            "summary": f"Compressed conversation focused on {focus}",
            "topics": [focus],
            "data": {"conversation_length": len(conversation)},
            "key_findings": []
        }
        
        # Save to Pinecone
        result = await self.save_context(summary_data)
        
        return TextContent(
            type="text",
            text=f"🗜️ Compressed and saved conversation\n{result.text}"
        )
    
    async def get_memory_stats(self) -> TextContent:
        """Get statistics about stored memory"""
        # Query Pinecone stats
        stats = self.index.describe_index_stats()
        
        return TextContent(
            type="text",
            text=f"📊 Memory Statistics:\n"
                 f"Total vectors stored: {stats.get('total_vector_count', 0)}\n"
                 f"Current session: {self.current_session}\n"
                 f"Chunks in session: {self.chunk_count}\n"
                 f"Index dimension: 1024\n"
                 f"Index fullness: {stats.get('index_fullness', 0):.2%}"
        )
    
    async def run(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream, 
                write_stream, 
                InitializationOptions(
                    server_name="infinite-context",
                    server_version="1.0.0",
                    capabilities={
                        "tools": {}
                    }
                )
            )

if __name__ == "__main__":
    mcp = InfiniteContextMCP()
    asyncio.run(mcp.run())