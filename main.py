# infinite_context_mcp.py
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ImageContent, EmbeddedResource
from mcp.server.models import InitializationOptions

import tiktoken
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Import query understanding engine with temporal awareness
from query_understanding import QueryUnderstandingEngine, RewriteType, TemporalScope, TemporalInfo

# Import indexing service
from lib.indexing_service import IndexingService, IndexingError, APIError

# Import user profile manager (Memory System)
from user_profile import UserProfileManager, UserProfile

# Import entity graph (Memory System - Knowledge Graph)
from entity_graph import EntityGraph, RelationType, Entity, Relationship

# Import fact chain manager (Memory System - Fact Chaining)
from fact_chain import FactChainManager, Fact, FactType, FactCluster

# Import memory scorer (Memory System - Hybrid Scoring)
from memory_scorer import MemoryScorer, ScoringWeights, ScoreBreakdown

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
        
        # Initialize Indexing Service
        self.indexing_service = IndexingService(self.index, self.openai)
        
        # Initialize User Profile Manager (Memory System)
        self.profile_manager = UserProfileManager(self.index, self.openai)
        
        # Initialize Entity Graph (Memory System - Knowledge Graph)
        self.entity_graph = EntityGraph(self.index, self.openai)
        
        # Initialize Fact Chain Manager (Memory System - Fact Chaining)
        self.fact_manager = FactChainManager(self.index, self.openai)
        
        # Initialize Memory Scorer (Memory System - Hybrid Scoring)
        self.memory_scorer = MemoryScorer(openai_client=self.openai)
        
        # Current user profile (loaded on first use)
        self._current_profile: Optional[UserProfile] = None
        
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
                            "key_findings": {"type": "array", "items": {"type": "string"}},
                            "content": {"type": "string", "description": "Full content to save (optional)"}
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
                    description="Enhanced search with query understanding, rewrites, guardrails, automatic refinement, and intelligent follow-up recommendations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to search for"},
                            "top_k": {"type": "integer", "default": 3},
                            "use_rewrites": {"type": "boolean", "default": True, "description": "Use query rewrites for better recall"},
                            "min_relevance": {"type": "number", "default": 0.7, "description": "Minimum relevance score threshold"},
                            "auto_refine": {"type": "boolean", "default": True, "description": "Automatically refine search with best rewrite if results are poor (default: True)"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="ask_question",
                    description="RAG Question Answering - Ask questions about your saved data. Searches relevant contexts and generates comprehensive answers based on your saved information.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question you want to ask about your saved data (e.g., 'what am I building?', 'what projects have I worked on?', 'what did I learn about Pinecone?')"
                            },
                            "top_k": {
                                "type": "integer",
                                "default": 10,
                                "description": "Number of relevant contexts to retrieve for answering"
                            },
                            "min_relevance": {
                                "type": "number",
                                "default": 0.3,
                                "description": "Minimum relevance threshold for including contexts"
                            }
                        },
                        "required": ["question"]
                    }
                ),
                Tool(
                    name="smart_action",
                    description="Intelligent orchestration tool that automatically routes requests and combines tools for frictionless context management. Just describe what you want in natural language. When saving conversations, if conversation_context is provided, it will automatically analyze and extract summary, topics, key findings, and decisions from the conversation.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "request": {
                                "type": "string",
                                "description": "Natural language request describing what you want to do (e.g., 'save this conversation about MCP setup', 'find past discussions about Pinecone', 'show me memory stats', 'search for context about query understanding')"
                            },
                            "conversation_context": {
                                "type": "string",
                                "description": "Optional: Full conversation context. When saving, this will be automatically analyzed to extract summary, topics, key findings, and decisions. If not provided, the system will infer from the request."
                            }
                        },
                        "required": ["request"]
                    }
                ),
                Tool(
                    name="index_repository",
                    description="Index a GitHub repository for intelligent code search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "repo_url": {"type": "string", "description": "GitHub repository URL (e.g., https://github.com/owner/repo)"},
                            "branch": {"type": "string", "description": "Branch to index (optional, defaults to main branch)"},
                            "file_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional file patterns to include (regex)"}
                        },
                        "required": ["repo_url"]
                    }
                ),
                Tool(
                    name="index_documentation",
                    description="Index a documentation site for intelligent search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL of the documentation site"},
                            "url_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional URL patterns to include"},
                            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional URL patterns to exclude"},
                            "only_main_content": {"type": "boolean", "description": "Extract only main content", "default": True}
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="index_website",
                    description="Index a full website by crawling",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Root URL of the website"},
                            "max_depth": {"type": "integer", "description": "Maximum crawl depth", "default": 3},
                            "max_pages": {"type": "integer", "description": "Maximum pages to index", "default": 100},
                            "url_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional URL patterns to include"},
                            "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional URL patterns to exclude"},
                            "only_main_content": {"type": "boolean", "description": "Extract only main content", "default": True},
                            "wait_for": {"type": "integer", "description": "Wait time in ms between requests"},
                            "include_screenshot": {"type": "boolean", "description": "Include screenshots", "default": False}
                        },
                        "required": ["url"]
                    }
                ),
                Tool(
                    name="index_local_filesystem",
                    description="Index a local filesystem directory",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "directory_path": {"type": "string", "description": "Absolute path to directory"},
                            "inclusion_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional patterns to include"},
                            "exclusion_patterns": {"type": "array", "items": {"type": "string"}, "description": "Optional patterns to exclude"},
                            "max_file_size_mb": {"type": "integer", "description": "Maximum file size in MB", "default": 50}
                        },
                        "required": ["directory_path"]
                    }
                ),
                Tool(
                    name="check_indexing_status",
                    description="Check the indexing status of a source",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string", "description": "Source ID returned from indexing"}
                        },
                        "required": ["source_id"]
                    }
                ),
                Tool(
                    name="list_indexed_sources",
                    description="List all indexed sources",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="delete_indexed_source",
                    description="Delete an indexed source",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string", "description": "Source ID to delete"}
                        },
                        "required": ["source_id"]
                    }
                ),
                Tool(
                    name="index_url",
                    description="Index a single URL (any type - ChatGPT conversations, Twitter posts, blog posts, etc.)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to index (e.g., ChatGPT conversation link, Twitter URL, blog post, etc.)"},
                            "only_main_content": {"type": "boolean", "description": "Extract only main content (remove navigation, ads, etc.)", "default": True},
                            "wait_for": {"type": "integer", "description": "Wait time in ms before fetching (for dynamic content)"}
                        },
                        "required": ["url"]
                    }
                ),
                # Memory System Tools
                Tool(
                    name="get_user_profile",
                    description="Get the current user profile with preferences, entities, and context. This shows what the system knows about you.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Optional user ID (defaults to current user)"}
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="update_user_profile",
                    description="Manually update user profile preferences or focus areas",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string", "description": "Optional user ID"},
                            "current_focus": {"type": "string", "description": "Set current focus/project"},
                            "add_topics": {"type": "array", "items": {"type": "string"}, "description": "Topics to add to interests"},
                            "preferences": {"type": "object", "description": "Key-value preferences to set"}
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="query_knowledge_graph",
                    description="Query the knowledge graph for entity relationships. Find connections between people, projects, concepts, etc.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity_name": {"type": "string", "description": "Name of entity to query"},
                            "find_path_to": {"type": "string", "description": "Optional: Find path to another entity"},
                            "relation_types": {"type": "array", "items": {"type": "string"}, "description": "Optional: Filter by relation types"},
                            "max_depth": {"type": "integer", "description": "Max traversal depth (default: 2)", "default": 2}
                        },
                        "required": ["entity_name"]
                    }
                ),
                Tool(
                    name="get_graph_summary",
                    description="Get a summary of the knowledge graph - entities, relationships, and most connected nodes",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                ),
                Tool(
                    name="query_facts",
                    description="Query extracted facts by entity, type, or get related facts. Facts are atomic pieces of knowledge extracted from saved contexts.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string", "description": "Get facts mentioning this entity"},
                            "fact_type": {"type": "string", "description": "Filter by fact type: statement, preference, decision, event, relationship, state, capability, requirement, problem, solution"},
                            "include_superseded": {"type": "boolean", "description": "Include superseded (outdated) facts", "default": False},
                            "limit": {"type": "integer", "description": "Maximum facts to return", "default": 20}
                        },
                        "required": []
                    }
                ),
                Tool(
                    name="get_fact_summary",
                    description="Get a summary of all extracted facts - counts by type, clusters, and recent facts",
                    inputSchema={
                        "type": "object",
                        "properties": {},
                        "required": []
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
            elif name == "ask_question":
                result = await self.ask_question(arguments)
                return [result]
            elif name == "smart_action":
                result = await self.smart_action(arguments)
                return [result]
            elif name == "index_repository":
                result = await self.index_repository(arguments)
                return [result]
            elif name == "index_documentation":
                result = await self.index_documentation(arguments)
                return [result]
            elif name == "index_website":
                result = await self.index_website(arguments)
                return [result]
            elif name == "index_local_filesystem":
                result = await self.index_local_filesystem(arguments)
                return [result]
            elif name == "check_indexing_status":
                result = await self.check_indexing_status(arguments)
                return [result]
            elif name == "list_indexed_sources":
                result = await self.list_indexed_sources(arguments)
                return [result]
            elif name == "delete_indexed_source":
                result = await self.delete_indexed_source(arguments)
                return [result]
            elif name == "index_url":
                result = await self.index_url(arguments)
                return [result]
            # Memory System tools
            elif name == "get_user_profile":
                result = await self.get_user_profile(arguments)
                return [result]
            elif name == "update_user_profile":
                result = await self.update_user_profile(arguments)
                return [result]
            elif name == "query_knowledge_graph":
                result = await self.query_knowledge_graph(arguments)
                return [result]
            elif name == "get_graph_summary":
                result = await self.get_graph_summary(arguments)
                return [result]
            elif name == "query_facts":
                result = await self.query_facts(arguments)
                return [result]
            elif name == "get_fact_summary":
                result = await self.get_fact_summary(arguments)
                return [result]
            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    async def _get_current_profile(self, user_id: Optional[str] = None) -> UserProfile:
        """Get or create the current user profile"""
        if self._current_profile is None or (user_id and self._current_profile.user_id != user_id):
            self._current_profile = self.profile_manager.get_or_create_profile(user_id)
        return self._current_profile
    
    async def save_context(self, args: dict) -> TextContent:
        """Save context to Pinecone with Memory System integration"""
        # Ensure topics and key_findings are always lists (handle None values)
        topics = args.get('topics') or []
        key_findings = args.get('key_findings') or []
        data = args.get('data') or {}
        content = args.get('content') or ''
        user_id = args.get('user_id')  # Optional user ID
        
        # Get temporal validity fields (Memory System feature)
        valid_from = args.get('valid_from', datetime.now().isoformat())
        valid_until = args.get('valid_until')  # None means indefinitely valid
        supersedes = args.get('supersedes')  # ID of chunk this supersedes
        
        # Get current user profile
        profile = await self._get_current_profile(user_id)
        
        # Extract entities from content for profile update
        extracted_entities = {}
        if content:
            extracted_entities = await self.profile_manager.extract_entities_from_content(content)
        
        # Generate embedding
        # Note: text-embedding-3-large has a token limit. We include content but truncate it for embedding generation
        # to avoid API errors, while storing a larger chunk in metadata.
        text = f"""
        Summary: {args.get('summary', '')}
        Topics: {', '.join(topics)}
        Findings: {', '.join(key_findings)}
        Data: {json.dumps(data)}
        Content: {content[:20000]}
        """
        
        response = self.openai.embeddings.create(
            input=text[:30000], # Hard truncate to be safe for embedding model
            model="text-embedding-3-large",
            dimensions=1024
        )
        
        embedding = response.data[0].embedding
        
        # Store in Pinecone
        chunk_id = f"{self.current_session}_chunk_{self.chunk_count}"
        
        # Pinecone metadata limit is 40KB per vector. We allocate:
        # - Content: ~25KB
        # - Summary: ~2KB
        # - Data: ~2KB
        # - Others: ~1KB
        
        # Build metadata with Memory System fields
        metadata = {
                "session_id": self.current_session,
                "chunk_id": self.chunk_count,
                "timestamp": datetime.now().isoformat(),
                "summary": args.get('summary', '')[:2000],
                "topics": str(topics),
                "findings": str(key_findings),
                "data": json.dumps(data)[:2000],
            "content": content[:25000],
            # Memory System fields
            "user_id": profile.user_id,
            "valid_from": valid_from,
            "valid_until": valid_until or "",
            "supersedes": supersedes or "",
            # Entity data for relationship tracking
            "entities": json.dumps(extracted_entities)[:2000]
        }
        
        self.index.upsert([{
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata
        }])
        
        self.chunk_count += 1
        
        # Update user profile with new context (Memory System feature)
        await self.profile_manager.update_profile_from_context(
            profile=profile,
            content=content,
            summary=args.get('summary', ''),
            topics=topics,
            extracted_entities=extracted_entities
        )
        
        # Save updated profile
        await self.profile_manager.save_profile(profile)
        
        # Extract entities and relationships for knowledge graph (Memory System feature)
        graph_entities = []
        graph_relationships = []
        extracted_facts = []
        
        if content and len(content) > 100:  # Only process substantial content
            self.entity_graph.set_user(profile.user_id)
            graph_entities, graph_relationships = await self.entity_graph.extract_entities_and_relationships(
                content=content,
                source_chunk_id=chunk_id
            )
            
            # Extract and chain facts (Memory System feature)
            self.fact_manager.set_user(profile.user_id)
            entity_names = [e.name for e in graph_entities] if graph_entities else None
            extracted_facts = await self.fact_manager.extract_facts(
                content=content,
                source_chunk_id=chunk_id,
                existing_entities=entity_names
            )
            
            # Save graph and facts periodically (every 5 chunks)
            if self.chunk_count % 5 == 0:
                await self.entity_graph.save_graph()
                await self.fact_manager.save_facts()
        
        # Build response
        entity_info = ""
        if extracted_entities:
            entity_counts = {k: len(v) for k, v in extracted_entities.items() if v}
            if entity_counts:
                entity_info = f"\n🔗 Entities extracted: {entity_counts}"
        
        graph_info = ""
        if graph_entities or graph_relationships:
            graph_info = f"\n🕸️ Knowledge graph: {len(graph_entities)} entities, {len(graph_relationships)} relationships"
        
        fact_info = ""
        if extracted_facts:
            fact_info = f"\n📝 Facts extracted: {len(extracted_facts)}"
        
        return TextContent(
            type="text",
            text=f"✅ Saved context chunk {chunk_id}\n"
                 f"Session: {self.current_session}\n"
                 f"Topics: {', '.join(topics) if topics else 'None'}\n"
                 f"Findings: {len(key_findings)} key findings stored"
                 f"{entity_info}"
                 f"{graph_info}"
                 f"{fact_info}\n"
                 f"👤 Profile updated: {profile.total_contexts} total contexts"
        )
    
    async def search_context(self, args: dict) -> TextContent:
        """
        Smart Memory Search - Uses user profiles for default context injection
        and QueryUnderstandingEngine to understand user intent.
        This is the main search method that combines memory system with RAG.
        """
        query = args.get('query', '')
        top_k = args.get('top_k', 5)  # Reduced for speed
        min_relevance = args.get('min_relevance', 0.3)
        generate_ai = args.get('generate_ai_response', True)  # Can disable for faster results
        user_id = args.get('user_id')  # Optional user ID
        use_profile_context = args.get('use_profile_context', True)  # Memory System feature
        
        if not query:
            return TextContent(
                type="text",
                text="❌ Please provide a search query."
            )
        
        # Step 0: Get user profile for default context (Memory System feature)
        profile = await self._get_current_profile(user_id)
        profile_context = ""
        if use_profile_context and profile.profile_summary:
            profile_context = self.profile_manager.get_profile_context(profile)
        
        # Step 1: Fast query classification (cached, non-blocking)
        classification = self.query_engine.classify_query(query)
        query_type_str = classification.query_type.value if hasattr(classification.query_type, 'value') else str(classification.query_type)
        
        # Step 2: Generate single embedding for main query (fast)
        # Optionally include profile context in embedding for better personalization
        query_for_embedding = query
        if profile_context and use_profile_context:
            # Include user context in embedding for personalized search
            query_for_embedding = f"{profile_context}\n\nQuery: {query}"
        
        response = self.openai.embeddings.create(
            input=query_for_embedding[:8000],
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding
        
        # Step 3: Search Pinecone with user filtering (Memory System feature)
        # Filter by user_id if profile exists
        filter_dict = None
        if profile.user_id and profile.user_id != "default_user":
            filter_dict = {"user_id": {"$eq": profile.user_id}}
        
        results = self.index.query(
            vector=embedding,
            top_k=top_k * 2,  # Get more results for better context
            include_metadata=True,
            filter=filter_dict
        )
        
        all_results = results['matches']
        
        # If no user-specific results, fall back to all results
        if not all_results and filter_dict:
            results = self.index.query(
                vector=embedding,
                top_k=top_k * 2,
                include_metadata=True
            )
            all_results = results['matches']
        
        # Step 4: Apply Temporal Filtering (Memory System feature)
        temporal_info = classification.temporal_info
        if temporal_info and temporal_info.scope != TemporalScope.ALL_TIME:
            all_results = self.query_engine.apply_temporal_filter(all_results, temporal_info)
        
        # Step 5: Build context for hybrid scoring
        # Prepare profile context
        profile_scoring_context = None
        if profile:
            profile_scoring_context = {
                'recent_topics': profile.temporal_context.recent_topics or [],
                'entities': [e.name for e in list(profile.entities.values())[:20]],
                'current_focus': profile.temporal_context.current_focus or ''
            }
        
        # Prepare temporal context
        temporal_scoring_context = None
        if temporal_info:
            temporal_scoring_context = {
                'scope': temporal_info.scope.value,
                'start_date': temporal_info.start_date.isoformat() if temporal_info.start_date else None,
                'end_date': temporal_info.end_date.isoformat() if temporal_info.end_date else None
            }
        
        # Prepare entity context from knowledge graph
        entity_scoring_context = None
        if self.entity_graph._entities:
            # Get entities related to query
            query_entities = [w for w in query.split() if len(w) > 2 and w[0].isupper()]
            related_entities = []
            for qe in query_entities:
                entity = self.entity_graph.get_entity_by_name(qe)
                if entity:
                    related = self.entity_graph.get_related_entities(entity.id, max_depth=1)
                    related_entities.extend([e.name for e, r in related])
            
            entity_scoring_context = {
                'query_entities': query_entities,
                'related_entities': list(set(related_entities))[:20]
            }
        
        # Prepare fact context
        fact_scoring_context = None
        if self.fact_manager._facts:
            # Get relevant facts for the query
            relevant_facts = []
            for word in query.lower().split():
                if len(word) > 3:
                    facts = self.fact_manager.get_facts_by_entity(word, current_only=True)
                    relevant_facts.extend([{'source_chunk_id': f.source_chunk_id} for f in facts[:5]])
            
            fact_scoring_context = {
                'relevant_facts': relevant_facts[:20]
            }
        
        # Step 6: Apply Advanced Hybrid Scoring (Memory System)
        # Adjust weights based on query type
        adjusted_weights = self.memory_scorer.adjust_weights_for_query_type(query_type_str)
        self.memory_scorer.weights = adjusted_weights
        
        # Rank results with full memory system scoring
        scored_results = self.memory_scorer.rank_results(
            results=all_results,
            query=query,
            profile_context=profile_scoring_context,
            temporal_context=temporal_scoring_context,
            entity_context=entity_scoring_context,
            fact_context=fact_scoring_context
        )
        
        # Extract just the results
        all_results = [r for r, breakdown in scored_results]
        
        # Step 7: Apply threshold filtering
        if query_type_str == 'question':
            effective_min_relevance = min(min_relevance, 0.2)
        else:
            effective_min_relevance = min_relevance
        
        final_results = [r for r in all_results if r.get('hybrid_score', r.get('score', 0)) >= effective_min_relevance][:top_k]
        
        # If no results, take top ones anyway
        if not final_results and all_results:
            final_results = all_results[:top_k]
        
        # Step 8: Generate AI Answer (RAG-style) with profile context - Memory System enhanced
        result_text = ""
        
        # Show temporal context if detected (Memory System feature)
        if temporal_info and temporal_info.temporal_keywords:
            result_text += f"⏰ **Temporal Context:** {temporal_info.scope.value}"
            if temporal_info.start_date:
                result_text += f" (from {temporal_info.start_date.strftime('%Y-%m-%d')})"
            result_text += "\n\n"
        
        # Show profile context if available (Memory System feature)
        if profile_context and use_profile_context:
            result_text += f"👤 **User Context Applied**\n{profile.profile_summary[:200]}...\n\n"
        
        if final_results:
            if generate_ai:
                # Pass profile context to AI response generation
                ai_answer = await self.generate_ai_response(query, final_results, profile_context)
                result_text += f"🤖 **Answer:**\n\n{ai_answer}\n\n"
                result_text += "---\n\n"
            
            result_text += f"📚 **Sources ({len(final_results)}):**\n\n"
            
            for i, match in enumerate(final_results[:5], 1):
                meta = match['metadata']
                summary = meta.get('summary', 'Untitled')[:100]
                timestamp = meta.get('timestamp', 'N/A')[:10] if meta.get('timestamp') else 'N/A'
                score = match.get('hybrid_score', match.get('score', 0))
                
                # Show score breakdown if available
                breakdown = match.get('score_breakdown', {})
                result_text += f"**[{i}]** {summary}...\n"
                result_text += f"   📊 {score:.2%}"
                
                # Show top contributing factors
                if breakdown.get('components'):
                    components = breakdown['components']
                    top_factors = sorted(
                        [(k, v) for k, v in components.items() if v > 0.5],
                        key=lambda x: x[1],
                        reverse=True
                    )[:2]
                    if top_factors:
                        factor_strs = [f"{k[:4]}:{v:.0%}" for k, v in top_factors]
                        result_text += f" ({', '.join(factor_strs)})"
                
                result_text += f" | 📅 {timestamp}\n\n"
            
            if len(final_results) > 5:
                result_text += f"... and {len(final_results) - 5} more\n"
        else:
            result_text += f"❌ **No relevant contexts found**\n\n"
            result_text += f"💡 Try rephrasing your question or using more general terms.\n"
        
        return TextContent(type="text", text=result_text)
    
    async def generate_ai_response(self, query: str, search_results: list, profile_context: str = "") -> str:
        """Generate a fast AI response based on search results with profile context (Memory System)"""
        if not search_results:
            return "I couldn't find any relevant information in your saved contexts."
        
        # Build concise context from top 2 results for speed
        context_pieces = []
        for i, match in enumerate(search_results[:2], 1):  # Use top 2 for speed
            meta = match.get('metadata', {})
            # Prefer summary, fallback to content snippet (shorter)
            content = meta.get('summary', '') or meta.get('content', '')[:300]  # Even shorter
            if content:
                context_pieces.append(f"[{i}] {content}")
        
        # Build prompt with profile context (Memory System feature)
        profile_section = ""
        if profile_context:
            profile_section = f"User Context:\n{profile_context}\n\n"
        
        # Ultra-concise prompt for fastest generation
        user_prompt = f"""{profile_section}Q: {query}\n\n{chr(10).join(context_pieces)}\n\nAnswer briefly. Cite [1], [2]. Consider user context if relevant."""
        
        try:
            # Optimized for speed: single message, minimal tokens, low temperature
            completion = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": user_prompt}  # Single message for speed
                ],
                temperature=0.2,  # Very low for fastest responses
                max_tokens=400  # Reduced further for speed
            )
            
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    async def ask_question(self, args: dict) -> TextContent:
        """
        RAG Question Answering - Ask questions about your saved data.
        Searches relevant contexts and generates comprehensive answers.
        """
        question = args.get('question', '')
        top_k = args.get('top_k', 10)  # Get more contexts for better answers
        min_relevance = args.get('min_relevance', 0.3)  # Lower threshold for broader context
        
        if not question:
            return TextContent(
                type="text",
                text="❌ Please provide a question to answer."
            )
        
        # Step 1: Search for relevant contexts using enhanced search
        search_args = {
            'query': question,
            'top_k': top_k,
            'use_rewrites': True,
            'min_relevance': min_relevance,
            'generate_ai_response': False,  # We'll generate our own
            'auto_refine': True
        }
        
        # Get search results
        search_result = await self.enhanced_search(search_args)
        
        # Extract results from the search response
        # Parse the search result text to get the actual matches
        search_text = search_result.text
        
        # Use the search method directly to get structured results
        response = self.openai.embeddings.create(
            input=question,
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
        
        # Apply hybrid scoring
        query_keywords = set(question.lower().split())
        all_results = []
        for match in results['matches']:
            meta = match.get('metadata', {})
            content_text = ' '.join([
                str(meta.get('summary', '')),
                str(meta.get('content', '')),
                str(meta.get('topics', '')),
            ]).lower()
            
            keyword_matches = sum(1 for keyword in query_keywords if keyword in content_text)
            keyword_score = keyword_matches / max(len(query_keywords), 1)
            exact_phrase_bonus = 0.2 if question.lower() in content_text else 0
            
            match['hybrid_score'] = (
                0.7 * match.get('score', 0) + 
                0.2 * keyword_score + 
                exact_phrase_bonus
            )
            match['keyword_matches'] = keyword_matches
            all_results.append(match)
        
        # Sort by hybrid score and filter
        all_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        filtered_results = [r for r in all_results if r.get('hybrid_score', 0) >= min_relevance]
        
        # Step 2: Generate comprehensive AI answer
        if not filtered_results:
            return TextContent(
                type="text",
                text=f"❓ **Question:** {question}\n\n"
                     f"❌ I couldn't find any relevant information in your saved contexts to answer this question.\n\n"
                     f"💡 **Suggestions:**\n"
                     f"- Try rephrasing your question\n"
                     f"- Use more general terms\n"
                     f"- Make sure you have saved relevant context first"
            )
        
        ai_answer = await self.generate_ai_response(question, filtered_results[:5])
        
        # Step 3: Format the response
        response_text = f"❓ **Question:** {question}\n\n"
        response_text += f"🤖 **Answer:**\n\n{ai_answer}\n\n"
        response_text += "---\n\n"
        response_text += f"📚 **Sources ({len(filtered_results)} contexts used):**\n\n"
        
        for i, match in enumerate(filtered_results[:5], 1):
            meta = match['metadata']
            summary = meta.get('summary', 'Untitled')[:80]
            timestamp = meta.get('timestamp', 'N/A')[:10] if meta.get('timestamp') else 'N/A'
            score = match.get('hybrid_score', match.get('score', 0))
            response_text += f"[{i}] {summary}... (relevance: {score:.2%})\n"
            if meta.get('topics'):
                topics = str(meta.get('topics', ''))[:50]
                response_text += f"    Topics: {topics}\n"
            response_text += f"    Date: {timestamp}\n\n"
        
        if len(filtered_results) > 5:
            response_text += f"... and {len(filtered_results) - 5} more contexts\n"
        
        return TextContent(type="text", text=response_text)
    
    async def search_with_ai_response(self, args: dict) -> TextContent:
        """Search context and generate an AI response like Perplexity"""
        query = args.get('query', '')
        top_k = args.get('top_k', 5)
        
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
        
        # Generate AI response
        ai_response = await self.generate_ai_response(query, results['matches'])
        
        # Format the complete response
        response_text = f"🤖 **AI Response**\n\n{ai_response}\n\n"
        response_text += "---\n\n📚 **Sources:**\n\n"
        
        for i, match in enumerate(results['matches'], 1):
            meta = match['metadata']
            response_text += f"[{i}] {meta.get('summary', 'Untitled context')[:100]}... (relevance: {match['score']:.2%})\n"
        
        return TextContent(type="text", text=response_text)
    
    async def classify_query(self, args: dict) -> TextContent:
        """Classify a query to understand intent and categories"""
        query = args.get('query', '')
        context = args.get('context')
        
        classification = self.query_engine.classify_query(query, context)
        
        result_text = f"🎯 Query Classification for: '{query}'\n\n"
        query_type_str = classification.query_type.value if hasattr(classification.query_type, 'value') else str(classification.query_type)
        result_text += f"Query Type: {query_type_str}\n"
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
    
    async def _generate_followup_recommendations(
        self, 
        original_query: str, 
        classification, 
        rewrites: List, 
        results: List[Dict],
        all_results: List[Dict]
    ) -> List[str]:
        """
        Generate follow-up query recommendations based on search results and generated rewrites.
        """
        recommendations = []
        
        # If no results or poor results, recommend trying rewrites
        if len(results) == 0 or (results and results[0].get('score', 0) < 0.6):
            if rewrites:
                recommendations.append(f"💡 Try a different approach: '{rewrites[0].rewrite}' (from generated rewrites)")
                if len(rewrites) > 1:
                    recommendations.append(f"💡 Or try: '{rewrites[1].rewrite}'")
        
        # Recommend broader/narrower searches based on results
        if results:
            # Extract topics from results
            result_topics = set()
            for result in results[:3]:
                meta = result.get('metadata', {})
                topics_str = meta.get('topics', '')
                if topics_str:
                    try:
                        topics = eval(topics_str) if isinstance(topics_str, str) else topics_str
                        if isinstance(topics, list):
                            result_topics.update(topics)
                    except:
                        pass
            
            if result_topics:
                topics_list = list(result_topics)[:3]
                recommendations.append(f"🔍 Explore related topics: '{', '.join(topics_list)}'")
            
            # If results are good, suggest narrowing
            if results[0].get('score', 0) > 0.8:
                recommendations.append(f"🎯 Narrow your search: Add more specific terms to '{original_query}'")
        else:
            # No results - suggest broader search
            recommendations.append(f"🔍 Try a broader search: Remove specific terms from '{original_query}'")
            if classification.categories:
                recommendations.append(f"📂 Search by category: '{classification.categories[0]}'")
        
        # Use LLM to generate intelligent recommendations
        try:
            results_summary = f"Found {len(results)} results. " + \
                            (f"Top result relevance: {results[0].get('score', 0):.1%}" if results else "No results found.")
            
            prompt = f"""Based on this search scenario, suggest 2-3 follow-up queries that would help the user find what they're looking for.

Original Query: "{original_query}"
Query Intent: {classification.intent}
Query Categories: {', '.join(classification.categories[:3]) if classification.categories else 'None'}
Generated Rewrites: {', '.join([r.rewrite for r in rewrites[:3]]) if rewrites else 'None'}
Search Results: {results_summary}

Suggest follow-up queries that:
1. Try alternative phrasings if no results
2. Narrow down if too many results
3. Explore related topics found in results
4. Use different search angles

Respond with JSON array of query strings:
{{"recommendations": ["query 1", "query 2", "query 3"]}}
"""

            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a search query expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            llm_recs = json.loads(response.choices[0].message.content)
            recommendations.extend(llm_recs.get("recommendations", [])[:3])
        except Exception as e:
            pass  # Fallback to basic recommendations
        
        return recommendations[:5]  # Limit to 5 recommendations
    
    def _generate_typo_tolerant_queries(self, query: str) -> list:
        """Generate typo-tolerant variations of the query"""
        variations = []
        
        # Common typo patterns
        typo_map = {
            'pinecone': ['pincone', 'pinecode', 'pine cone'],
            'raycast': ['raycat', 'ray cast', 'raycost'],
            'search': ['serach', 'seach', 'searchh'],
            'context': ['contex', 'conext', 'contextt'],
            'query': ['querry', 'qeury', 'queery'],
            'understanding': ['understnading', 'understaning', 'undestanding'],
            'mcp': ['mpc', 'mc p', 'mcpp'],
            'api': ['ap i', 'apii', 'aip'],
            'server': ['sever', 'servr', 'servre'],
            'config': ['confg', 'conifg', 'configuration']
        }
        
        # Check if query contains common typos
        query_lower = query.lower()
        for correct, typos in typo_map.items():
            for typo in typos:
                if typo in query_lower:
                    # Generate corrected version
                    corrected = query_lower.replace(typo, correct)
                    variations.append(corrected)
        
        # Also check reverse - if query has correct word, add common typos
        # This helps find content where the typo was in the saved context
        for correct, typos in typo_map.items():
            if correct in query_lower:
                for typo in typos[:1]:  # Just one typo variant
                    typo_version = query_lower.replace(correct, typo)
                    variations.append(typo_version)
        
        return variations[:3]  # Limit variations
    
    async def enhanced_search(self, args: dict) -> TextContent:
        """
        Enhanced search with query understanding, rewrites, and guardrails.
        Implements the Instacart-inspired approach.
        """
        query = args.get('query', '')
        top_k = args.get('top_k', 3)
        use_rewrites = args.get('use_rewrites', True)
        min_relevance = args.get('min_relevance', 0.7)
        auto_refine = args.get('auto_refine', True)  # Default: automatically refine if poor results
        
        result_text = f"🚀 Enhanced Search for: '{query}'\n\n"
        
        # Step 1: Classify the query
        classification = self.query_engine.classify_query(query)
        result_text += f"📊 Query Classification:\n"
        query_type_str = classification.query_type.value if hasattr(classification.query_type, 'value') else str(classification.query_type)
        result_text += f"   Type: {query_type_str}\n"
        result_text += f"   Intent: {classification.intent}\n"
        result_text += f"   Categories: {', '.join(classification.categories) if classification.categories else 'None'}\n\n"
        
        # Step 2: Query Expansion - Generate multiple search strategies
        search_queries = [query]
        rewrites = []
        
        # Extract key entities and concepts from the query
        # This helps with typo tolerance and concept matching
        query_tokens = query.lower().split()
        
        if use_rewrites:
            # Generate semantic rewrites
            rewrites = self.query_engine.generate_rewrites(
                query, 
                [RewriteType.SYNONYM, RewriteType.EXPANSION, RewriteType.BROADER, RewriteType.SUBSTITUTE]
            )
            
            # Add spelling corrections and common variations
            # This helps when users mistype or use different terminology
            if len(query_tokens) <= 5:  # Only for shorter queries
                # Add common typo corrections
                typo_variations = self._generate_typo_tolerant_queries(query)
                search_queries.extend(typo_variations[:2])
            
            # Add top semantic rewrites
            top_rewrites = sorted(rewrites, key=lambda x: x.confidence, reverse=True)[:3]
            for rewrite in top_rewrites:
                search_queries.append(rewrite.rewrite)
            
            result_text += f"✏️ Query Expansion: {len(search_queries)} search variants\n"
            for i, sq in enumerate(search_queries[:4], 1):  # Show first 4
                result_text += f"   {i}. {sq}\n"
            if len(search_queries) > 4:
                result_text += f"   ... and {len(search_queries) - 4} more\n"
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
            
            # Add unique results with query-specific scoring
            for match in results['matches']:
                chunk_id = match['metadata'].get('chunk_id')
                if chunk_id not in seen_ids:
                    # Add query source for hybrid scoring
                    match['query_source'] = search_query
                    all_results.append(match)
                    seen_ids.add(chunk_id)
        
        # Step 3.5: Apply Hybrid Scoring (Semantic + Keyword)
        # This improves relevance by combining vector similarity with keyword matching
        query_keywords = set(query.lower().split())
        for result in all_results:
            meta = result.get('metadata', {})
            
            # Calculate keyword match score
            content_text = ' '.join([
                str(meta.get('summary', '')),
                str(meta.get('content', '')),
                str(meta.get('topics', '')),
                str(meta.get('findings', ''))
            ]).lower()
            
            # Count keyword matches
            keyword_matches = sum(1 for keyword in query_keywords if keyword in content_text)
            keyword_score = keyword_matches / max(len(query_keywords), 1)
            
            # Check for exact phrase match (bonus score)
            exact_phrase_bonus = 0.2 if query.lower() in content_text else 0
            
            # Combine scores (70% semantic, 20% keyword, 10% exact phrase)
            original_score = result.get('score', 0)
            result['hybrid_score'] = (
                0.7 * original_score + 
                0.2 * keyword_score + 
                exact_phrase_bonus
            )
            result['keyword_matches'] = keyword_matches
        
        # Sort by hybrid score
        all_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        
        # Step 4: Apply guardrails using hybrid scores
        # Update scores for guardrail evaluation
        for result in all_results:
            result['score'] = result.get('hybrid_score', result.get('score', 0))
        
        filtered_results, filter_reasons = self.query_engine.apply_guardrails(
            query, all_results, min_relevance
        )
        
        # Sort by hybrid score and take top_k
        filtered_results.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)
        final_results = filtered_results[:top_k]
        
        result_text += f"🛡️ Guardrails Applied:\n"
        result_text += f"   Results before filtering: {len(all_results)}\n"
        result_text += f"   Results after filtering: {len(final_results)}\n"
        if filter_reasons:
            result_text += f"   Filtered out: {len(filter_reasons)} results\n"
        result_text += "\n"
        
        # Step 5: Generate AI Response (Perplexity-style)
        if final_results and args.get('generate_ai_response', True):
            result_text += f"🤖 **AI Response:**\n\n"
            ai_response = await self.generate_ai_response(query, final_results)
            result_text += f"{ai_response}\n\n"
            result_text += "---\n\n"
        
        # Step 6: Display results
        result_text += f"📄 **Search Results ({len(final_results)}):**\n\n"
        
        if not final_results:
            result_text += "No results found matching the relevance threshold.\n"
        else:
            for i, match in enumerate(final_results, 1):
                meta = match['metadata']
                result_text += f"**[{i}]** {meta.get('summary', 'Untitled')[:100]}...\n"
                result_text += f"   📊 Score: {match.get('hybrid_score', match['score']):.2%}"
                if match.get('keyword_matches'):
                    result_text += f" (🔤 {match['keyword_matches']} keywords)"
                result_text += f" | 📅 {meta.get('timestamp', 'N/A')[:10]}\n"
                if meta.get('topics'):
                    result_text += f"   🏷️  Topics: {meta.get('topics', 'N/A')}\n"
                result_text += "\n"
        
        # Step 6: Auto-refine if poor results and enabled
        if auto_refine and (not final_results or (final_results and final_results[0].get('score', 0) < 0.6)):
            if rewrites:
                result_text += f"🔄 Auto-refining search with best rewrite...\n\n"
                # Try the best rewrite
                best_rewrite = sorted(rewrites, key=lambda x: x.confidence, reverse=True)[0]
                refined_result = await self.enhanced_search({
                    "query": best_rewrite.rewrite,
                    "top_k": top_k,
                    "use_rewrites": False,  # Don't rewrite again
                    "min_relevance": min_relevance,
                    "auto_refine": False  # Prevent infinite loop
                })
                result_text += f"✨ Refined Search Results:\n{refined_result.text}\n"
        
        # Step 7: Generate follow-up recommendations
        recommendations = await self._generate_followup_recommendations(
            query, classification, rewrites, final_results, all_results
        )
        
        if recommendations:
            result_text += "\n" + "=" * 60 + "\n"
            result_text += "💡 Follow-up Query Recommendations:\n\n"
            for i, rec in enumerate(recommendations, 1):
                result_text += f"   {i}. {rec}\n"
            result_text += "\n"
            result_text += "💬 Tip: Use these queries with enhanced_search for better results!\n"
        
        # Add cache stats
        cache_stats = self.query_engine.get_cache_stats()
        result_text += f"\n💾 Cache Stats: {cache_stats['hit_rate']} hit rate\n"
        
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
        
        # Get user profile stats (Memory System)
        profile = await self._get_current_profile()
        
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
                 f"   Hit rate: {cache_stats['hit_rate']}\n\n"
                 f"👤 User Profile (Memory System):\n"
                 f"   User ID: {profile.user_id}\n"
                 f"   Total contexts: {profile.total_contexts}\n"
                 f"   Total facts: {profile.total_facts}\n"
                 f"   Known entities: {len(profile.entities)}\n"
                 f"   Topic interests: {len(profile.topic_interests)}"
        )
    
    async def get_user_profile(self, args: dict) -> TextContent:
        """Get detailed user profile information (Memory System)"""
        user_id = args.get('user_id')
        profile = await self._get_current_profile(user_id)
        
        result_text = f"👤 **User Profile: {profile.user_id}**\n\n"
        
        # Profile summary
        if profile.profile_summary:
            result_text += f"📝 **Summary:**\n{profile.profile_summary}\n\n"
        
        # Temporal context
        result_text += "⏰ **Current Context:**\n"
        if profile.temporal_context.current_focus:
            result_text += f"   Focus: {profile.temporal_context.current_focus}\n"
        if profile.temporal_context.recent_topics:
            result_text += f"   Recent topics: {', '.join(profile.temporal_context.recent_topics[:5])}\n"
        if profile.temporal_context.last_interaction:
            result_text += f"   Last interaction: {profile.temporal_context.last_interaction[:19]}\n"
        result_text += "\n"
        
        # Top interests
        if profile.topic_interests:
            top_interests = sorted(profile.topic_interests.items(), key=lambda x: x[1], reverse=True)[:10]
            result_text += "🎯 **Top Interests:**\n"
            for topic, score in top_interests:
                bar = "█" * int(score * 10)
                result_text += f"   {topic}: {bar} ({score:.1%})\n"
            result_text += "\n"
        
        # Key entities
        if profile.entities:
            result_text += "🔗 **Known Entities:**\n"
            by_type: Dict[str, List[str]] = {}
            for entity in profile.entities.values():
                if entity.entity_type not in by_type:
                    by_type[entity.entity_type] = []
                by_type[entity.entity_type].append(f"{entity.name} ({entity.mention_count})")
            
            for entity_type, entities in by_type.items():
                result_text += f"   {entity_type.title()}: {', '.join(entities[:5])}\n"
            result_text += "\n"
        
        # Statistics
        result_text += "📊 **Statistics:**\n"
        result_text += f"   Total contexts saved: {profile.total_contexts}\n"
        result_text += f"   Total facts extracted: {profile.total_facts}\n"
        result_text += f"   Profile created: {profile.created_at[:10]}\n"
        result_text += f"   Last updated: {profile.updated_at[:10]}\n"
        
        return TextContent(type="text", text=result_text)
    
    async def update_user_profile(self, args: dict) -> TextContent:
        """Manually update user profile (Memory System)"""
        user_id = args.get('user_id')
        profile = await self._get_current_profile(user_id)
        
        changes = []
        
        # Update current focus
        if args.get('current_focus'):
            profile.temporal_context.current_focus = args['current_focus']
            changes.append(f"Set current focus to: {args['current_focus']}")
        
        # Add topics to interests
        if args.get('add_topics'):
            for topic in args['add_topics']:
                topic_lower = topic.lower()
                current_score = profile.topic_interests.get(topic_lower, 0.0)
                profile.topic_interests[topic_lower] = min(1.0, current_score + 0.2)
            changes.append(f"Added topics: {', '.join(args['add_topics'])}")
        
        # Update preferences
        if args.get('preferences'):
            from user_profile import UserPreference
            now = datetime.now().isoformat()
            for key, value in args['preferences'].items():
                pref_key = f"manual:{key}"
                profile.preferences[pref_key] = UserPreference(
                    category="manual",
                    key=key,
                    value=value,
                    confidence=1.0,
                    last_updated=now
                )
            changes.append(f"Updated preferences: {list(args['preferences'].keys())}")
        
        # Regenerate profile summary
        profile.profile_summary = await self.profile_manager._generate_profile_summary(profile)
        
        # Save profile
        await self.profile_manager.save_profile(profile)
        
        result_text = f"✅ **Profile Updated: {profile.user_id}**\n\n"
        if changes:
            result_text += "Changes made:\n"
            for change in changes:
                result_text += f"  • {change}\n"
        else:
            result_text += "No changes specified.\n"
        
        result_text += f"\n📝 New summary: {profile.profile_summary[:200]}..."
        
        return TextContent(type="text", text=result_text)
    
    async def query_knowledge_graph(self, args: dict) -> TextContent:
        """Query the knowledge graph for entity relationships (Memory System)"""
        entity_name = args.get('entity_name', '')
        find_path_to = args.get('find_path_to')
        relation_types_str = args.get('relation_types', [])
        max_depth = args.get('max_depth', 2)
        
        if not entity_name:
            return TextContent(type="text", text="❌ Please provide an entity name to query.")
        
        # Get current profile to set user context
        profile = await self._get_current_profile()
        self.entity_graph.set_user(profile.user_id)
        
        # Try to load graph if not loaded
        if not self.entity_graph._entities:
            await self.entity_graph.load_graph(profile.user_id)
        
        # Find the entity
        entity = self.entity_graph.get_entity_by_name(entity_name)
        
        if not entity:
            return TextContent(
                type="text",
                text=f"❌ Entity '{entity_name}' not found in knowledge graph.\n\n"
                     f"💡 Try saving more context to build the knowledge graph, or check the entity name."
            )
        
        result_text = f"🕸️ **Knowledge Graph Query: {entity.name}**\n\n"
        result_text += f"**Entity Type:** {entity.entity_type}\n"
        result_text += f"**First Seen:** {entity.first_seen[:10] if entity.first_seen else 'N/A'}\n"
        result_text += f"**Mentions:** {entity.mention_count}\n"
        
        if entity.attributes:
            result_text += f"**Attributes:** {json.dumps(entity.attributes)}\n"
        
        result_text += "\n"
        
        # If finding path to another entity
        if find_path_to:
            target_entity = self.entity_graph.get_entity_by_name(find_path_to)
            if target_entity:
                path_result = self.entity_graph.traverse_path(
                    entity.id, target_entity.id, max_depth=max_depth
                )
                
                if path_result:
                    result_text += f"**🔗 Path to {find_path_to}:**\n"
                    result_text += f"   {path_result.explanation}\n"
                    result_text += f"   Strength: {path_result.total_strength:.2%}\n\n"
                else:
                    result_text += f"**❌ No path found to {find_path_to}** (within {max_depth} hops)\n\n"
            else:
                result_text += f"**❌ Target entity '{find_path_to}' not found**\n\n"
        
        # Get related entities
        relation_types = None
        if relation_types_str:
            try:
                relation_types = [RelationType(rt) for rt in relation_types_str]
            except:
                pass
        
        related = self.entity_graph.get_related_entities(
            entity.id, 
            relation_types=relation_types,
            max_depth=max_depth
        )
        
        if related:
            result_text += f"**🔗 Related Entities ({len(related)}):**\n\n"
            
            # Group by relationship type
            by_relation: Dict[str, List[Tuple[Entity, Relationship]]] = {}
            for rel_entity, rel in related:
                rel_type = rel.relation_type.value
                if rel_type not in by_relation:
                    by_relation[rel_type] = []
                by_relation[rel_type].append((rel_entity, rel))
            
            for rel_type, items in by_relation.items():
                result_text += f"**{rel_type.replace('_', ' ').title()}:**\n"
                for rel_entity, rel in items[:5]:
                    strength_bar = "█" * int(rel.strength * 5)
                    result_text += f"   • {rel_entity.name} ({rel_entity.entity_type}) {strength_bar}\n"
                if len(items) > 5:
                    result_text += f"   ... and {len(items) - 5} more\n"
                result_text += "\n"
        else:
            result_text += "**No related entities found.**\n"
        
        return TextContent(type="text", text=result_text)
    
    async def get_graph_summary(self, args: dict) -> TextContent:
        """Get knowledge graph summary (Memory System)"""
        # Get current profile to set user context
        profile = await self._get_current_profile()
        self.entity_graph.set_user(profile.user_id)
        
        # Try to load graph if not loaded
        if not self.entity_graph._entities:
            await self.entity_graph.load_graph(profile.user_id)
        
        summary = self.entity_graph.get_graph_summary()
        
        result_text = f"🕸️ **Knowledge Graph Summary**\n\n"
        result_text += f"**Total Entities:** {summary['total_entities']}\n"
        result_text += f"**Total Relationships:** {summary['total_relationships']}\n\n"
        
        if summary['entity_types']:
            result_text += "**Entity Types:**\n"
            for entity_type, count in sorted(summary['entity_types'].items(), key=lambda x: x[1], reverse=True):
                result_text += f"   • {entity_type}: {count}\n"
            result_text += "\n"
        
        if summary['relation_types']:
            result_text += "**Relationship Types:**\n"
            for rel_type, count in sorted(summary['relation_types'].items(), key=lambda x: x[1], reverse=True):
                result_text += f"   • {rel_type}: {count}\n"
            result_text += "\n"
        
        if summary['most_connected']:
            result_text += "**Most Connected Entities:**\n"
            for item in summary['most_connected']:
                result_text += f"   • {item['name']} ({item['type']}): {item['connections']} connections\n"
        
        if summary['total_entities'] == 0:
            result_text += "\n💡 The knowledge graph is empty. Save more context to build it."
        
        return TextContent(type="text", text=result_text)
    
    async def query_facts(self, args: dict) -> TextContent:
        """Query extracted facts (Memory System)"""
        entity = args.get('entity')
        fact_type_str = args.get('fact_type')
        include_superseded = args.get('include_superseded', False)
        limit = args.get('limit', 20)
        
        # Get current profile to set user context
        profile = await self._get_current_profile()
        self.fact_manager.set_user(profile.user_id)
        
        # Try to load facts if not loaded
        if not self.fact_manager._facts:
            await self.fact_manager.load_facts(profile.user_id)
        
        facts = []
        
        if entity:
            # Get facts by entity
            facts = self.fact_manager.get_facts_by_entity(entity, current_only=not include_superseded)
            result_text = f"📝 **Facts about '{entity}'**\n\n"
        elif fact_type_str:
            # Get facts by type
            try:
                fact_type = FactType(fact_type_str)
                facts = self.fact_manager.get_facts_by_type(fact_type, current_only=not include_superseded)
                result_text = f"📝 **Facts of type '{fact_type_str}'**\n\n"
            except ValueError:
                return TextContent(
                    type="text",
                    text=f"❌ Invalid fact type: {fact_type_str}\n\n"
                         f"Valid types: statement, preference, decision, event, relationship, state, capability, requirement, problem, solution"
                )
        else:
            # Get all current facts
            facts = self.fact_manager.get_current_facts(limit=limit)
            result_text = f"📝 **Recent Facts**\n\n"
        
        if not facts:
            result_text += "No facts found.\n\n"
            result_text += "💡 Save more context to extract facts."
            return TextContent(type="text", text=result_text)
        
        # Display facts
        facts = facts[:limit]
        result_text += f"Found {len(facts)} facts:\n\n"
        
        for i, fact in enumerate(facts, 1):
            status = "✅" if fact.is_current else "❌ (superseded)"
            result_text += f"**{i}. [{fact.fact_type.value}]** {status}\n"
            result_text += f"   {fact.content}\n"
            if fact.entities:
                result_text += f"   Entities: {', '.join(fact.entities[:5])}\n"
            result_text += f"   Confidence: {fact.confidence:.0%} | {fact.source_timestamp[:10] if fact.source_timestamp else 'N/A'}\n"
            if fact.superseded_by_fact_id:
                result_text += f"   ⚠️ Superseded by newer fact\n"
            result_text += "\n"
        
        return TextContent(type="text", text=result_text)
    
    async def get_fact_summary(self, args: dict) -> TextContent:
        """Get fact store summary (Memory System)"""
        # Get current profile to set user context
        profile = await self._get_current_profile()
        self.fact_manager.set_user(profile.user_id)
        
        # Try to load facts if not loaded
        if not self.fact_manager._facts:
            await self.fact_manager.load_facts(profile.user_id)
        
        summary = self.fact_manager.get_summary()
        
        result_text = f"📝 **Fact Store Summary**\n\n"
        result_text += f"**Total Facts:** {summary['total_facts']}\n"
        result_text += f"**Current Facts:** {summary['current_facts']}\n"
        result_text += f"**Superseded Facts:** {summary['superseded_facts']}\n"
        result_text += f"**Fact Clusters:** {summary['clusters']}\n"
        result_text += f"**Entities Tracked:** {summary['entities_tracked']}\n\n"
        
        if summary['fact_types']:
            result_text += "**Facts by Type:**\n"
            for fact_type, count in sorted(summary['fact_types'].items(), key=lambda x: x[1], reverse=True):
                result_text += f"   • {fact_type}: {count}\n"
            result_text += "\n"
        
        # Show recent facts
        recent_facts = self.fact_manager.get_current_facts(limit=5)
        if recent_facts:
            result_text += "**Recent Facts:**\n"
            for fact in recent_facts:
                result_text += f"   • [{fact.fact_type.value}] {fact.content[:80]}...\n"
        
        if summary['total_facts'] == 0:
            result_text += "\n💡 The fact store is empty. Save more context to extract facts."
        
        return TextContent(type="text", text=result_text)
    
    async def _analyze_conversation(self, conversation: str, user_request: str = "") -> dict:
        """
        Analyze conversation context to extract summary, topics, key findings, and decisions.
        Uses LLM to intelligently extract information from the conversation.
        """
        if not conversation or len(conversation.strip()) < 50:
            return {
                "summary": user_request[:200] if user_request else "Conversation context",
                "topics": [],
                "key_findings": [],
                "data": {}
            }
        
        analysis_prompt = f"""Analyze the following conversation and extract key information for context storage.

Conversation:
{conversation[:8000]}  # Limit to avoid token limits

User's save request: "{user_request}"

Extract and provide:
1. A concise summary (2-3 sentences) of what was discussed
2. Main topics/themes (3-7 topics as a list)
3. Key findings, decisions, or important points (3-10 bullet points)
4. Any structured data, facts, or metrics mentioned (as JSON object)

Respond in JSON format:
{{
    "summary": "Brief summary of the conversation",
    "topics": ["topic1", "topic2", "topic3"],
    "key_findings": [
        "Finding 1",
        "Finding 2",
        "Finding 3"
    ],
    "data": {{
        "key": "value",
        "metric": 123
    }}
}}

Focus on:
- What was actually discussed/accomplished
- Important decisions made
- Key insights or discoveries
- Technical details, configurations, or facts
- Problems solved or solutions found
"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing conversations and extracting key information. Always respond with valid JSON."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            analysis = json.loads(response.choices[0].message.content)
            
            # Ensure all fields exist
            return {
                "summary": analysis.get("summary", user_request[:200] if user_request else "Conversation context"),
                "topics": analysis.get("topics", [])[:10],  # Limit topics
                "key_findings": analysis.get("key_findings", [])[:15],  # Limit findings
                "data": analysis.get("data", {})
            }
        except Exception as e:
            # Fallback to basic extraction
            classification = self.query_engine.classify_query(conversation[:500] if conversation else user_request)
            return {
                "summary": conversation[:300] if conversation else user_request[:200],
                "topics": classification.categories[:5] if classification.categories else [],
                "key_findings": [],
                "data": {}
            }
    
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
Conversation Context: "{conversation_context[:1000] if conversation_context else 'None'}"

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
                # If conversation_context is provided, analyze it to extract information
                if conversation_context and len(conversation_context.strip()) > 50:
                    result_text += "🧠 Analyzing conversation context...\n\n"
                    analysis = await self._analyze_conversation(conversation_context, request)
                    
                    # Use analyzed data, but allow user-provided parameters to override
                    if not parameters.get("summary"):
                        parameters["summary"] = analysis["summary"]
                    if not parameters.get("topics") or len(parameters.get("topics", [])) == 0:
                        parameters["topics"] = analysis["topics"]
                    if not parameters.get("key_findings") or len(parameters.get("key_findings", [])) == 0:
                        parameters["key_findings"] = analysis["key_findings"]
                    if not parameters.get("data") or len(parameters.get("data", {})) == 0:
                        parameters["data"] = analysis["data"]
                    
                    # Ensure full content is saved
                    if not parameters.get("content"):
                        parameters["content"] = conversation_context
                    
                    result_text += f"✅ Extracted:\n"
                    result_text += f"   📝 Summary: {analysis['summary'][:100]}...\n"
                    result_text += f"   🏷️  Topics: {', '.join(analysis['topics'][:5])}\n"
                    result_text += f"   🔍 Key Findings: {len(analysis['key_findings'])} items\n"
                    result_text += "\n"
                else:
                    # Fallback: Extract or infer parameters from request
                    if not parameters.get("summary"):
                        parameters["summary"] = request[:200] if request else "Conversation context"
                    if not parameters.get("topics"):
                        # Classify to get topics
                        classification = self.query_engine.classify_query(request)
                        parameters["topics"] = classification.categories[:5] if classification.categories else []
                    if not parameters.get("key_findings"):
                        parameters["key_findings"] = []
                    if not parameters.get("data"):
                        parameters["data"] = {}
                    if not parameters.get("content"):
                        parameters["content"] = request # Use request as content if no conversation
                
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
                if "auto_refine" not in parameters:
                    parameters["auto_refine"] = True  # Default to True, automatically refine
                
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
    
    async def index_repository(self, args: dict) -> TextContent:
        """Index a GitHub repository"""
        try:
            repo_url = args.get('repo_url', '')
            branch = args.get('branch')
            file_patterns = args.get('file_patterns')
            
            result = await self.indexing_service.index_repository(
                repo_url=repo_url,
                branch=branch,
                file_patterns=file_patterns
            )
            
            return TextContent(
                type="text",
                text=f"✅ Successfully indexed repository: {result['repository']}\n"
                     f"Branch: {result['branch']}\n"
                     f"Files indexed: {result['files_indexed']}\n"
                     f"Chunks created: {result['chunks_created']}\n"
                     f"Source ID: {result['source_id']}"
            )
        except APIError as e:
            error_msg = f"❌ {str(e)}"
            if e.status_code == 403:
                error_msg += "\n\n💡 Tip: Check your GitHub token permissions or rate limits."
            return TextContent(type="text", text=error_msg)
        except IndexingError as e:
            return TextContent(type="text", text=f"❌ Error indexing repository: {str(e)}")
        except Exception as e:
            return TextContent(type="text", text=f"❌ Unexpected error: {str(e)}")
    
    async def index_documentation(self, args: dict) -> TextContent:
        """Index a documentation site"""
        try:
            url = args.get('url', '')
            url_patterns = args.get('url_patterns')
            exclude_patterns = args.get('exclude_patterns')
            only_main_content = args.get('only_main_content', True)
            
            result = await self.indexing_service.index_documentation(
                url=url,
                url_patterns=url_patterns,
                exclude_patterns=exclude_patterns,
                only_main_content=only_main_content
            )
            
            return TextContent(
                type="text",
                text=f"✅ Successfully indexed documentation: {url}\n"
                     f"Pages indexed: {result['pages_indexed']}\n"
                     f"Chunks created: {result['chunks_created']}\n"
                     f"Source ID: {result['source_id']}"
            )
        except IndexingError as e:
            return TextContent(type="text", text=f"❌ Error indexing documentation: {str(e)}")
        except Exception as e:
            return TextContent(type="text", text=f"❌ Unexpected error: {str(e)}")
    
    async def index_website(self, args: dict) -> TextContent:
        """Index a full website"""
        try:
            url = args.get('url', '')
            max_depth = args.get('max_depth', 3)
            max_pages = args.get('max_pages', 100)
            url_patterns = args.get('url_patterns')
            exclude_patterns = args.get('exclude_patterns')
            only_main_content = args.get('only_main_content', True)
            wait_for = args.get('wait_for')
            include_screenshot = args.get('include_screenshot', False)
            
            result = await self.indexing_service.index_website(
                url=url,
                max_depth=max_depth,
                max_pages=max_pages,
                url_patterns=url_patterns,
                exclude_patterns=exclude_patterns,
                only_main_content=only_main_content,
                wait_for=wait_for,
                include_screenshot=include_screenshot
            )
            
            return TextContent(
                type="text",
                text=f"✅ Successfully indexed website: {url}\n"
                     f"Pages indexed: {result['pages_indexed']}\n"
                     f"Chunks created: {result['chunks_created']}\n"
                     f"Source ID: {result['source_id']}"
            )
        except IndexingError as e:
            return TextContent(type="text", text=f"❌ Error indexing website: {str(e)}")
        except Exception as e:
            return TextContent(type="text", text=f"❌ Unexpected error: {str(e)}")
    
    async def index_local_filesystem(self, args: dict) -> TextContent:
        """Index a local filesystem directory"""
        try:
            directory_path = args.get('directory_path', '')
            inclusion_patterns = args.get('inclusion_patterns')
            exclusion_patterns = args.get('exclusion_patterns')
            max_file_size_mb = args.get('max_file_size_mb', 50)
            
            result = await self.indexing_service.index_local_filesystem(
                directory_path=directory_path,
                inclusion_patterns=inclusion_patterns,
                exclusion_patterns=exclusion_patterns,
                max_file_size_mb=max_file_size_mb
            )
            
            return TextContent(
                type="text",
                text=f"✅ Successfully indexed directory: {directory_path}\n"
                     f"Files indexed: {result['files_indexed']}\n"
                     f"Chunks created: {result['chunks_created']}\n"
                     f"Source ID: {result['source_id']}"
            )
        except IndexingError as e:
            return TextContent(type="text", text=f"❌ Error indexing filesystem: {str(e)}")
        except Exception as e:
            return TextContent(type="text", text=f"❌ Unexpected error: {str(e)}")
    
    async def check_indexing_status(self, args: dict) -> TextContent:
        """Check indexing status"""
        try:
            source_id = args.get('source_id', '')
            status = self.indexing_service.get_indexing_status(source_id)
            
            if not status:
                return TextContent(
                    type="text",
                    text=f"❌ Source ID '{source_id}' not found."
                )
            
            status_icon = {
                "completed": "✅",
                "indexing": "⏳",
                "failed": "❌",
                "deleted": "🗑️"
            }.get(status.get("status", "unknown"), "❓")
            
            lines = [
                f"{status_icon} **Status:** {status.get('status', 'unknown')}",
                f"**Source Type:** {status.get('source_type', 'unknown')}",
                f"**Source URL:** {status.get('source_url', 'N/A')}"
            ]
            
            if status.get("progress") is not None:
                lines.append(f"**Progress:** {status['progress']}%")
            
            if status.get("page_count"):
                lines.append(f"**Pages/Files:** {status['page_count']}")
            
            if status.get("chunk_count"):
                lines.append(f"**Chunks:** {status['chunk_count']}")
            
            if status.get("started_at"):
                lines.append(f"**Started:** {status['started_at']}")
            
            if status.get("completed_at"):
                lines.append(f"**Completed:** {status['completed_at']}")
            
            if status.get("error"):
                lines.append(f"**Error:** {status['error']}")
            
            return TextContent(type="text", text="\n".join(lines))
        except Exception as e:
            return TextContent(type="text", text=f"❌ Error checking status: {str(e)}")
    
    async def list_indexed_sources(self, args: dict) -> TextContent:
        """List all indexed sources"""
        try:
            sources = self.indexing_service.list_indexed_sources()
            
            if not sources:
                return TextContent(
                    type="text",
                    text="No indexed sources found.\n\nUse indexing tools to index repositories, websites, or directories."
                )
            
            lines = ["# Indexed Sources\n"]
            
            for source in sources:
                status_icon = {
                    "completed": "✅",
                    "indexing": "⏳",
                    "failed": "❌",
                    "deleted": "🗑️"
                }.get(source.get("status", "unknown"), "❓")
                
                lines.append(f"\n## {status_icon} {source.get('source_type', 'unknown').upper()}")
                lines.append(f"- **URL/Path:** {source.get('source_url', 'N/A')}")
                lines.append(f"- **Status:** {source.get('status', 'unknown')}")
                
                if source.get("progress") is not None:
                    lines.append(f"- **Progress:** {source['progress']}%")
                
                if source.get("page_count"):
                    lines.append(f"- **Pages/Files:** {source['page_count']}")
                
                if source.get("chunk_count"):
                    lines.append(f"- **Chunks:** {source['chunk_count']}")
            
            return TextContent(type="text", text="\n".join(lines))
        except Exception as e:
            return TextContent(type="text", text=f"❌ Error listing sources: {str(e)}")
    
    async def delete_indexed_source(self, args: dict) -> TextContent:
        """Delete an indexed source"""
        try:
            source_id = args.get('source_id', '')
            success = await self.indexing_service.delete_indexed_source(source_id)
            
            if success:
                return TextContent(
                    type="text",
                    text=f"✅ Successfully deleted source: {source_id}"
                )
            else:
                return TextContent(
                    type="text",
                    text=f"❌ Failed to delete source: {source_id}"
                )
        except Exception as e:
            return TextContent(type="text", text=f"❌ Error deleting source: {str(e)}")
    
    async def index_url(self, args: dict) -> TextContent:
        """Index a single URL (any type)"""
        try:
            url = args.get('url', '')
            only_main_content = args.get('only_main_content', True)
            wait_for = args.get('wait_for')
            
            result = await self.indexing_service.index_url(
                url=url,
                only_main_content=only_main_content,
                wait_for=wait_for
            )
            
            title_info = f"\nTitle: {result.get('title', 'N/A')}" if result.get('title') else ""
            
            return TextContent(
                type="text",
                text=f"✅ Successfully indexed URL: {url}{title_info}\n"
                     f"Chunks created: {result['chunks_created']}\n"
                     f"Content length: {result['content_length']:,} characters\n"
                     f"Source ID: {result['source_id']}"
            )
        except APIError as e:
            error_msg = f"❌ {str(e)}"
            if e.status_code == 403:
                error_msg += "\n\n💡 Tip: The URL may require authentication or be rate-limited."
            elif e.status_code == 404:
                error_msg += "\n\n💡 Tip: The URL may not exist or be accessible."
            return TextContent(type="text", text=error_msg)
        except IndexingError as e:
            return TextContent(type="text", text=f"❌ Error indexing URL: {str(e)}")
        except Exception as e:
            return TextContent(type="text", text=f"❌ Unexpected error: {str(e)}")
    
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