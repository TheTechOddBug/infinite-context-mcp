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

# Import query understanding engine
from query_understanding import QueryUnderstandingEngine, RewriteType

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
        
        # Initialize Query Understanding Engine (Instacart-inspired)
        self.query_engine = QueryUnderstandingEngine(self.openai)
        
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
                ),
                Tool(
                    name="classify_query",
                    description="Classify a query to understand its intent and categories (Instacart-inspired QU)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query to classify"},
                            "context": {"type": "object", "description": "Optional domain context"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="rewrite_query",
                    description="Generate query rewrites (synonyms, broader terms, expansions) to improve recall",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The query to rewrite"},
                            "rewrite_types": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["synonym", "broader", "expansion", "substitute"]},
                                "description": "Types of rewrites to generate"
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="enhanced_search",
                    description="Enhanced search with query understanding, rewrites, and guardrails",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "top_k": {"type": "integer", "default": 3},
                            "use_rewrites": {"type": "boolean", "default": True, "description": "Use query rewrites for better recall"},
                            "min_relevance": {"type": "number", "default": 0.7, "description": "Minimum relevance score threshold"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="smart_action",
                    description="Intelligent orchestration tool that automatically routes requests and combines tools for frictionless context management. Just describe what you want in natural language.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "request": {
                                "type": "string",
                                "description": "Natural language request describing what you want to do (e.g., 'save this conversation about MCP setup', 'find past discussions about Pinecone', 'show me memory stats', 'search for context about query understanding')"
                            },
                            "conversation_context": {
                                "type": "string",
                                "description": "Optional: Current conversation context to help understand the request better"
                            }
                        },
                        "required": ["request"]
                    }
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
            elif name == "classify_query":
                result = await self.classify_query(arguments)
                return [result]
            elif name == "rewrite_query":
                result = await self.rewrite_query(arguments)
                return [result]
            elif name == "enhanced_search":
                result = await self.enhanced_search(arguments)
                return [result]
            elif name == "smart_action":
                result = await self.smart_action(arguments)
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
    
    async def classify_query(self, args: dict) -> TextContent:
        """Classify a query to understand intent and categories"""
        query = args.get('query', '')
        context = args.get('context')
        
        classification = self.query_engine.classify_query(query, context)
        
        result_text = f"🎯 Query Classification for: '{query}'\n\n"
        result_text += f"Query Type: {classification.query_type.value}\n"
        result_text += f"Confidence: {classification.confidence:.1%}\n"
        result_text += f"Intent: {classification.intent}\n"
        result_text += f"Categories: {', '.join(classification.categories) if classification.categories else 'None'}\n"
        
        return TextContent(type="text", text=result_text)
    
    async def rewrite_query(self, args: dict) -> TextContent:
        """Generate query rewrites for better recall"""
        query = args.get('query', '')
        rewrite_types_input = args.get('rewrite_types', ['synonym', 'broader', 'expansion'])
        
        # Convert string types to RewriteType enum
        rewrite_types = []
        type_map = {
            'synonym': RewriteType.SYNONYM,
            'broader': RewriteType.BROADER,
            'expansion': RewriteType.EXPANSION,
            'substitute': RewriteType.SUBSTITUTE
        }
        for rt_str in rewrite_types_input:
            if rt_str in type_map:
                rewrite_types.append(type_map[rt_str])
        
        if not rewrite_types:
            rewrite_types = [RewriteType.SYNONYM, RewriteType.BROADER, RewriteType.EXPANSION]
        
        rewrites = self.query_engine.generate_rewrites(query, rewrite_types)
        
        result_text = f"✏️ Query Rewrites for: '{query}'\n\n"
        result_text += f"Generated {len(rewrites)} rewrites:\n\n"
        
        for i, rewrite in enumerate(rewrites, 1):
            result_text += f"{i}. [{rewrite.rewrite_type.value.upper()}] {rewrite.rewrite}\n"
            result_text += f"   Confidence: {rewrite.confidence:.1%}\n"
            result_text += f"   Reasoning: {rewrite.reasoning}\n\n"
        
        return TextContent(type="text", text=result_text)
    
    async def enhanced_search(self, args: dict) -> TextContent:
        """
        Enhanced search with query understanding, rewrites, and guardrails.
        Implements the Instacart-inspired approach.
        """
        query = args.get('query', '')
        top_k = args.get('top_k', 3)
        use_rewrites = args.get('use_rewrites', True)
        min_relevance = args.get('min_relevance', 0.7)
        
        result_text = f"🚀 Enhanced Search for: '{query}'\n\n"
        
        # Step 1: Classify the query
        classification = self.query_engine.classify_query(query)
        result_text += f"📊 Query Classification:\n"
        result_text += f"   Type: {classification.query_type.value}\n"
        result_text += f"   Intent: {classification.intent}\n"
        result_text += f"   Categories: {', '.join(classification.categories) if classification.categories else 'None'}\n\n"
        
        # Step 2: Generate rewrites if enabled
        search_queries = [query]
        if use_rewrites:
            rewrites = self.query_engine.generate_rewrites(
                query, 
                [RewriteType.SYNONYM, RewriteType.EXPANSION]
            )
            # Add top rewrites to search queries
            top_rewrites = sorted(rewrites, key=lambda x: x.confidence, reverse=True)[:2]
            for rewrite in top_rewrites:
                search_queries.append(rewrite.rewrite)
            
            result_text += f"✏️ Query Rewrites Generated: {len(rewrites)}\n"
            for i, rewrite in enumerate(top_rewrites, 1):
                result_text += f"   {i}. {rewrite.rewrite} ({rewrite.rewrite_type.value})\n"
            result_text += "\n"
        
        # Step 3: Search with all queries and aggregate results
        all_results = []
        seen_ids = set()
        
        for search_query in search_queries:
            # Generate embedding
            response = self.openai.embeddings.create(
                input=search_query,
                model="text-embedding-3-large",
                dimensions=1024
            )
            embedding = response.data[0].embedding
            
            # Search Pinecone
            results = self.index.query(
                vector=embedding,
                top_k=top_k * 2,  # Get more results to account for deduplication
                include_metadata=True
            )
            
            # Add unique results
            for match in results['matches']:
                chunk_id = match['metadata'].get('chunk_id')
                if chunk_id not in seen_ids:
                    all_results.append(match)
                    seen_ids.add(chunk_id)
        
        # Step 4: Apply guardrails
        filtered_results, filter_reasons = self.query_engine.apply_guardrails(
            query, all_results, min_relevance
        )
        
        # Sort by score and take top_k
        filtered_results.sort(key=lambda x: x.get('score', 0), reverse=True)
        final_results = filtered_results[:top_k]
        
        result_text += f"🛡️ Guardrails Applied:\n"
        result_text += f"   Results before filtering: {len(all_results)}\n"
        result_text += f"   Results after filtering: {len(final_results)}\n"
        if filter_reasons:
            result_text += f"   Filtered out: {len(filter_reasons)} results\n"
        result_text += "\n"
        
        # Step 5: Display results
        result_text += f"📄 Search Results ({len(final_results)}):\n\n"
        
        if not final_results:
            result_text += "No results found matching the relevance threshold.\n"
        else:
            for i, match in enumerate(final_results, 1):
                meta = match['metadata']
                result_text += f"📄 Result {i} (relevance: {match['score']:.2%})\n"
                result_text += f"   🆔 Chunk ID: {meta.get('chunk_id', 'N/A')}\n"
                result_text += f"   📅 Time: {meta.get('timestamp', 'N/A')}\n"
                result_text += f"   📝 Summary: {meta.get('summary', 'N/A')}\n"
                result_text += f"   🏷️  Topics: {meta.get('topics', 'N/A')}\n"
                if meta.get('findings'):
                    result_text += f"   🔍 Key Findings: {meta.get('findings', 'N/A')}\n"
                result_text += "\n"
        
        # Add cache stats
        cache_stats = self.query_engine.get_cache_stats()
        result_text += f"💾 Cache Stats: {cache_stats['hit_rate']} hit rate\n"
        
        return TextContent(type="text", text=result_text)
    
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
        
        # Get query engine cache stats
        cache_stats = self.query_engine.get_cache_stats()
        
        return TextContent(
            type="text",
            text=f"📊 Memory Statistics:\n"
                 f"Total vectors stored: {stats.get('total_vector_count', 0)}\n"
                 f"Current session: {self.current_session}\n"
                 f"Chunks in session: {self.chunk_count}\n"
                 f"Index dimension: 1024\n"
                 f"Index fullness: {stats.get('index_fullness', 0):.2%}\n\n"
                 f"🔍 Query Understanding Cache:\n"
                 f"   Cache size: {cache_stats['cache_size']}\n"
                 f"   Cache hits: {cache_stats['cache_hits']}\n"
                 f"   Cache misses: {cache_stats['cache_misses']}\n"
                 f"   Hit rate: {cache_stats['hit_rate']}"
        )
    
    async def smart_action(self, args: dict) -> TextContent:
        """
        Intelligent orchestration tool that automatically routes requests
        and combines tools for frictionless context management.
        """
        request = args.get('request', '')
        conversation_context = args.get('conversation_context', '')
        
        # Use LLM to understand intent and extract parameters
        orchestration_prompt = f"""You are an intelligent orchestration system for context management. 
Analyze the user's request and determine what action(s) to take.

Available tools:
1. save_context - Save conversation context (needs: summary, topics, key_findings, data)
2. search_context - Search past conversations (needs: query, top_k)
3. enhanced_search - Enhanced search with QU features (needs: query, top_k, use_rewrites, min_relevance)
4. get_memory_stats - Get memory statistics (no params needed)
5. classify_query - Classify query intent (needs: query)
6. rewrite_query - Generate query rewrites (needs: query, rewrite_types)
7. auto_compress - Compress conversation (needs: conversation, focus)

User Request: "{request}"
Conversation Context: "{conversation_context[:500] if conversation_context else 'None'}"

Determine:
1. Primary action (save_context, search_context, enhanced_search, get_memory_stats, classify_query, rewrite_query, auto_compress, or multi_action)
2. Extracted parameters as JSON
3. Whether to use enhanced_search vs basic search_context
4. Any additional actions needed

Respond in JSON format:
{{
    "action": "save_context|search_context|enhanced_search|get_memory_stats|classify_query|rewrite_query|auto_compress|multi_action",
    "parameters": {{}},
    "use_enhanced": true/false,
    "additional_actions": []
}}

Examples:
- "save this conversation about MCP setup" → {{"action": "save_context", "parameters": {{"summary": "...", "topics": ["MCP"]}}}}
- "find past discussions about Pinecone" → {{"action": "enhanced_search", "parameters": {{"query": "Pinecone discussions", "top_k": 5}}}}
- "show me memory stats" → {{"action": "get_memory_stats", "parameters": {{}}}}
- "search for context about query understanding and then save it" → {{"action": "multi_action", "parameters": {{"actions": [{{"action": "enhanced_search", ...}}, {{"action": "save_context", ...}}]}}}}
"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an intelligent orchestration system. Always respond with valid JSON."},
                    {"role": "user", "content": orchestration_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            action = plan.get("action", "search_context")
            parameters = plan.get("parameters", {})
            use_enhanced = plan.get("use_enhanced", True)
            additional_actions = plan.get("additional_actions", [])
            
            result_text = f"🎯 Smart Action: {request}\n\n"
            result_text += f"📋 Detected Action: {action}\n"
            result_text += f"⚙️  Parameters: {json.dumps(parameters, indent=2)}\n\n"
            result_text += "=" * 60 + "\n\n"
            
            # Execute primary action
            if action == "save_context":
                # Extract or infer parameters from request
                if not parameters.get("summary"):
                    # Try to infer from request
                    parameters["summary"] = request[:200]
                if not parameters.get("topics"):
                    # Classify to get topics
                    classification = self.query_engine.classify_query(request)
                    parameters["topics"] = classification.categories[:5] if classification.categories else []
                if not parameters.get("key_findings"):
                    parameters["key_findings"] = []
                if not parameters.get("data"):
                    parameters["data"] = {}
                
                result = await self.save_context(parameters)
                result_text += result.text
                
            elif action == "search_context":
                if not parameters.get("query"):
                    parameters["query"] = request
                if not parameters.get("top_k"):
                    parameters["top_k"] = 5
                
                result = await self.search_context(parameters)
                result_text += result.text
                
            elif action == "enhanced_search":
                if not parameters.get("query"):
                    parameters["query"] = request
                if not parameters.get("top_k"):
                    parameters["top_k"] = 5
                if "use_rewrites" not in parameters:
                    parameters["use_rewrites"] = True
                if "min_relevance" not in parameters:
                    parameters["min_relevance"] = 0.7
                
                result = await self.enhanced_search(parameters)
                result_text += result.text
                
            elif action == "get_memory_stats":
                result = await self.get_memory_stats()
                result_text += result.text
                
            elif action == "classify_query":
                if not parameters.get("query"):
                    parameters["query"] = request
                result = await self.classify_query(parameters)
                result_text += result.text
                
            elif action == "rewrite_query":
                if not parameters.get("query"):
                    parameters["query"] = request
                result = await self.rewrite_query(parameters)
                result_text += result.text
                
            elif action == "auto_compress":
                if not parameters.get("conversation"):
                    parameters["conversation"] = conversation_context or request
                if not parameters.get("focus"):
                    parameters["focus"] = "general"
                result = await self.auto_compress(parameters)
                result_text += result.text
                
            elif action == "multi_action":
                # Execute multiple actions in sequence
                actions = parameters.get("actions", [])
                for i, action_item in enumerate(actions, 1):
                    result_text += f"\n📌 Action {i}/{len(actions)}: {action_item.get('action', 'unknown')}\n"
                    result_text += "-" * 60 + "\n"
                    
                    action_name = action_item.get("action")
                    action_params = action_item.get("parameters", {})
                    
                    if action_name == "save_context":
                        result = await self.save_context(action_params)
                    elif action_name == "search_context":
                        result = await self.search_context(action_params)
                    elif action_name == "enhanced_search":
                        result = await self.enhanced_search(action_params)
                    elif action_name == "get_memory_stats":
                        result = await self.get_memory_stats()
                    elif action_name == "classify_query":
                        result = await self.classify_query(action_params)
                    elif action_name == "rewrite_query":
                        result = await self.rewrite_query(action_params)
                    else:
                        result = TextContent(type="text", text=f"Unknown action: {action_name}")
                    
                    result_text += result.text + "\n\n"
            else:
                # Default to enhanced search
                result = await self.enhanced_search({"query": request, "top_k": 5})
                result_text += result.text
            
            # Execute additional actions if any
            if additional_actions:
                result_text += "\n" + "=" * 60 + "\n"
                result_text += "📌 Additional Actions:\n\n"
                for i, add_action in enumerate(additional_actions, 1):
                    result_text += f"Action {i}: {add_action.get('action', 'unknown')}\n"
                    # Could execute these too if needed
            
            result_text += "\n" + "=" * 60 + "\n"
            result_text += "✅ Smart action completed!"
            
            return TextContent(type="text", text=result_text)
            
        except Exception as e:
            # Fallback: try enhanced search
            try:
                result = await self.enhanced_search({"query": request, "top_k": 5})
                return TextContent(
                    type="text",
                    text=f"🎯 Smart Action (Fallback): {request}\n\n"
                         f"⚠️ Orchestration parsing failed: {str(e)}\n"
                         f"Fell back to enhanced search:\n\n{result.text}"
                )
            except Exception as e2:
                return TextContent(
                    type="text",
                    text=f"❌ Error in smart_action: {str(e)}\nFallback also failed: {str(e2)}"
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