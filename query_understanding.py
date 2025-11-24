#!/usr/bin/env python3
"""
Query Understanding Module - Inspired by Instacart's Intent Engine
Implements query classification, rewrites, temporal awareness, and RAG-enhanced context retrieval
"""
import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

import openai
from dotenv import load_dotenv

load_dotenv()


class QueryType(Enum):
    """Types of queries we can classify"""
    SEARCH = "search"
    QUESTION = "question"
    COMMAND = "command"
    CONTEXT_RETRIEVAL = "context_retrieval"
    DATA_STORAGE = "data_storage"
    TEMPORAL = "temporal"  # New: time-based queries
    UNKNOWN = "unknown"


class TemporalScope(Enum):
    """Temporal scope for queries"""
    CURRENT = "current"  # Present state only
    HISTORICAL = "historical"  # Past states
    ALL_TIME = "all_time"  # No temporal filter
    SPECIFIC_RANGE = "specific_range"  # Specific date range


class RewriteType(Enum):
    """Types of query rewrites"""
    SYNONYM = "synonym"
    BROADER = "broader"
    SUBSTITUTE = "substitute"
    EXPANSION = "expansion"


@dataclass
class TemporalInfo:
    """Temporal information extracted from a query"""
    scope: TemporalScope
    reference_date: Optional[datetime] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    is_present_tense: bool = True
    temporal_keywords: List[str] = field(default_factory=list)


@dataclass
class QueryClassification:
    """Result of query classification with temporal awareness"""
    query_type: QueryType
    confidence: float
    categories: List[str]
    intent: str
    temporal_info: Optional[TemporalInfo] = None  # New: temporal context


@dataclass
class QueryRewrite:
    """A query rewrite suggestion"""
    rewrite: str
    rewrite_type: RewriteType
    confidence: float
    reasoning: str


class QueryUnderstandingEngine:
    """
    Query Understanding Engine inspired by Instacart's approach.
    Implements context-engineering, guardrails, temporal awareness, and query understanding.
    """
    
    # Temporal keyword patterns for detection
    PAST_KEYWORDS = [
        "was", "were", "had", "did", "used to", "previously", "before",
        "earlier", "last", "ago", "past", "former", "old", "historical",
        "back then", "at that time", "originally", "initially"
    ]
    
    PRESENT_KEYWORDS = [
        "is", "are", "has", "have", "do", "does", "currently", "now",
        "today", "present", "current", "active", "existing", "ongoing"
    ]
    
    FUTURE_KEYWORDS = [
        "will", "going to", "plan to", "intend to", "upcoming", "next",
        "future", "soon", "later", "tomorrow"
    ]
    
    RELATIVE_TIME_PATTERNS = {
        r"last\s+(\d+)\s+days?": lambda m: timedelta(days=int(m.group(1))),
        r"last\s+(\d+)\s+weeks?": lambda m: timedelta(weeks=int(m.group(1))),
        r"last\s+(\d+)\s+months?": lambda m: timedelta(days=int(m.group(1)) * 30),
        r"last\s+week": lambda m: timedelta(weeks=1),
        r"last\s+month": lambda m: timedelta(days=30),
        r"yesterday": lambda m: timedelta(days=1),
        r"today": lambda m: timedelta(days=0),
        r"this\s+week": lambda m: timedelta(days=7),
        r"this\s+month": lambda m: timedelta(days=30),
    }
    
    def __init__(self, openai_client: Optional[openai.OpenAI] = None):
        self.openai = openai_client or openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Domain-specific context for RAG (can be extended)
        self.domain_context = {
            "common_topics": [
                "MCP", "Pinecone", "Vector Storage", "Python", "Claude", 
                "Context Management", "Embeddings", "Semantic Search"
            ],
            "query_patterns": {
                "search": ["find", "search", "look for", "retrieve", "get"],
                "question": ["what", "how", "why", "when", "where", "explain"],
                "command": ["save", "store", "create", "delete", "update"],
                "context_retrieval": ["previous", "past", "earlier", "history", "remember"],
                "temporal": ["when", "since", "until", "before", "after", "during"]
            }
        }
        
        # Cache for frequent queries (hybrid approach)
        self.query_cache: Dict[str, Dict] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def detect_temporal_info(self, query: str) -> TemporalInfo:
        """
        Detect temporal information from a query.
        This enables time-aware fact retrieval.
        """
        query_lower = query.lower()
        now = datetime.now()
        
        # Detect temporal keywords
        found_past = [kw for kw in self.PAST_KEYWORDS if kw in query_lower]
        found_present = [kw for kw in self.PRESENT_KEYWORDS if kw in query_lower]
        found_future = [kw for kw in self.FUTURE_KEYWORDS if kw in query_lower]
        
        all_temporal_keywords = found_past + found_present + found_future
        
        # Determine scope
        scope = TemporalScope.ALL_TIME
        is_present_tense = True
        start_date = None
        end_date = None
        reference_date = now
        
        # Check for relative time patterns
        for pattern, delta_func in self.RELATIVE_TIME_PATTERNS.items():
            match = re.search(pattern, query_lower)
            if match:
                delta = delta_func(match)
                start_date = now - delta
                end_date = now
                scope = TemporalScope.SPECIFIC_RANGE
                all_temporal_keywords.append(match.group(0))
                break
        
        # If no specific range found, infer from keywords
        if scope == TemporalScope.ALL_TIME:
            if found_past and not found_present:
                scope = TemporalScope.HISTORICAL
                is_present_tense = False
            elif found_present and not found_past:
                scope = TemporalScope.CURRENT
                is_present_tense = True
        
        return TemporalInfo(
            scope=scope,
            reference_date=reference_date,
            start_date=start_date,
            end_date=end_date,
            is_present_tense=is_present_tense,
            temporal_keywords=all_temporal_keywords
        )
    
    def apply_temporal_filter(
        self, 
        results: List[Dict], 
        temporal_info: TemporalInfo
    ) -> List[Dict]:
        """
        Filter results based on temporal validity.
        This handles facts with validity windows.
        """
        if temporal_info.scope == TemporalScope.ALL_TIME:
            return results
        
        now = datetime.now()
        filtered = []
        
        for result in results:
            meta = result.get('metadata', {})
            
            # Get temporal validity from metadata
            valid_from_str = meta.get('valid_from', '')
            valid_until_str = meta.get('valid_until', '')
            
            # Parse dates
            valid_from = None
            valid_until = None
            
            if valid_from_str:
                try:
                    valid_from = datetime.fromisoformat(valid_from_str.replace('Z', '+00:00'))
                except:
                    pass
            
            if valid_until_str:
                try:
                    valid_until = datetime.fromisoformat(valid_until_str.replace('Z', '+00:00'))
                except:
                    pass
            
            # Apply temporal filtering based on scope
            include = True
            
            if temporal_info.scope == TemporalScope.CURRENT:
                # Only include facts that are currently valid
                if valid_until and valid_until < now:
                    include = False  # Fact has expired
                if valid_from and valid_from > now:
                    include = False  # Fact not yet valid
                    
            elif temporal_info.scope == TemporalScope.HISTORICAL:
                # Include facts that were valid in the past
                # This is more permissive - we want to see past states
                pass
                
            elif temporal_info.scope == TemporalScope.SPECIFIC_RANGE:
                # Check if fact was valid during the specified range
                if temporal_info.start_date and temporal_info.end_date:
                    # Fact should overlap with query range
                    if valid_until and valid_until < temporal_info.start_date:
                        include = False  # Fact expired before range
                    if valid_from and valid_from > temporal_info.end_date:
                        include = False  # Fact not valid until after range
            
            if include:
                # Add temporal relevance score
                result['temporal_relevance'] = self._calculate_temporal_relevance(
                    meta, temporal_info, now
                )
                filtered.append(result)
        
        return filtered
    
    def _calculate_temporal_relevance(
        self, 
        metadata: Dict, 
        temporal_info: TemporalInfo,
        now: datetime
    ) -> float:
        """Calculate temporal relevance score for a result"""
        relevance = 1.0
        
        # Get timestamp
        timestamp_str = metadata.get('timestamp', '')
        if not timestamp_str:
            return 0.5  # Neutral if no timestamp
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            return 0.5
        
        # Calculate recency bonus/penalty
        days_old = (now - timestamp).days
        
        if temporal_info.scope == TemporalScope.CURRENT:
            # Prefer recent facts for current queries
            if days_old <= 7:
                relevance = 1.0
            elif days_old <= 30:
                relevance = 0.8
            elif days_old <= 90:
                relevance = 0.6
            else:
                relevance = 0.4
                
        elif temporal_info.scope == TemporalScope.HISTORICAL:
            # For historical queries, older might be more relevant
            relevance = 0.8  # Slight penalty, but not much
            
        elif temporal_info.scope == TemporalScope.SPECIFIC_RANGE:
            # Check if timestamp is within range
            if temporal_info.start_date and temporal_info.end_date:
                if temporal_info.start_date <= timestamp <= temporal_info.end_date:
                    relevance = 1.0
                else:
                    relevance = 0.5
        
        return relevance
    
    def classify_query(self, query: str, context: Optional[Dict] = None) -> QueryClassification:
        """
        Classify a query into categories and determine intent.
        Uses RAG to inject domain context. Now includes temporal awareness.
        """
        # Detect temporal information first (fast, no API call)
        temporal_info = self.detect_temporal_info(query)
        
        # Check cache first (hybrid approach)
        cache_key = f"classify:{query.lower().strip()}"
        if cache_key in self.query_cache:
            self.cache_hits += 1
            cached = self.query_cache[cache_key]
            # Convert cached string query_type back to QueryType enum
            if isinstance(cached.get("query_type"), str):
                cached["query_type"] = QueryType(cached["query_type"])
            # Add temporal info (not cached as it depends on current time)
            cached["temporal_info"] = temporal_info
            return QueryClassification(**cached)
        
        self.cache_misses += 1
        
        # Build RAG-enhanced prompt with domain context
        domain_knowledge = self._get_relevant_domain_context(query, context)
        
        prompt = f"""You are a query understanding system. Classify the following query and determine its intent.

Domain Context:
{json.dumps(domain_knowledge, indent=2)}

Query: "{query}"

Analyze this query and provide:
1. Query Type: One of {[qt.value for qt in QueryType]}
2. Confidence: A score between 0.0 and 1.0
3. Categories: List of relevant categories/topics (max 5)
4. Intent: A brief description of what the user wants to accomplish

Respond in JSON format:
{{
    "query_type": "...",
    "confidence": 0.95,
    "categories": ["category1", "category2"],
    "intent": "description of user intent"
}}"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query classification expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Determine if this is a temporal query based on keywords
            query_type = result.get("query_type", "unknown")
            if temporal_info.temporal_keywords and query_type not in ["temporal"]:
                # Override to temporal if strong temporal signals
                if len(temporal_info.temporal_keywords) >= 2 or temporal_info.scope != TemporalScope.ALL_TIME:
                    query_type = "temporal"
            
            classification = QueryClassification(
                query_type=QueryType(query_type),
                confidence=float(result.get("confidence", 0.5)),
                categories=result.get("categories", []),
                intent=result.get("intent", ""),
                temporal_info=temporal_info
            )
            
            # Cache the result (without temporal_info as it's time-dependent)
            self.query_cache[cache_key] = {
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "categories": classification.categories,
                "intent": classification.intent
            }
            
            return classification
            
        except Exception as e:
            # Fallback classification
            return QueryClassification(
                query_type=QueryType.UNKNOWN,
                confidence=0.3,
                categories=[],
                intent=f"Error classifying query: {str(e)}",
                temporal_info=temporal_info
            )
    
    def generate_rewrites(
        self, 
        query: str, 
        rewrite_types: Optional[List[RewriteType]] = None,
        context: Optional[Dict] = None
    ) -> List[QueryRewrite]:
        """
        Generate query rewrites for better recall.
        Supports synonyms, broader queries, and substitutes.
        """
        if rewrite_types is None:
            rewrite_types = [RewriteType.SYNONYM, RewriteType.BROADER, RewriteType.EXPANSION]
        
        # Check cache
        cache_key = f"rewrite:{query.lower().strip()}:{','.join([rt.value for rt in rewrite_types])}"
        if cache_key in self.query_cache:
            self.cache_hits += 1
            cached = self.query_cache[cache_key]
            # Convert cached dicts back to QueryRewrite objects, converting string rewrite_type to enum
            rewrites = []
            for r in cached:
                rewrite_type_str = r.get("rewrite_type", "synonym")
                # Convert string to RewriteType enum
                rewrite_type_enum = RewriteType(rewrite_type_str) if isinstance(rewrite_type_str, str) else rewrite_type_str
                rewrites.append(QueryRewrite(
                    rewrite=r.get("rewrite", ""),
                    rewrite_type=rewrite_type_enum,
                    confidence=float(r.get("confidence", 0.5)),
                    reasoning=r.get("reasoning", "")
                ))
            return rewrites
        
        self.cache_misses += 1
        
        domain_knowledge = self._get_relevant_domain_context(query, context)
        
        rewrite_instructions = {
            RewriteType.SYNONYM: "Generate synonyms or alternative phrasings that mean the same thing",
            RewriteType.BROADER: "Generate broader, more general queries that encompass the original",
            RewriteType.EXPANSION: "Expand the query with related terms to improve recall",
            RewriteType.SUBSTITUTE: "Generate alternative queries that could substitute for the original"
        }
        
        instructions = "\n".join([
            f"- {rt.value.upper()}: {rewrite_instructions[rt]}"
            for rt in rewrite_types
        ])
        
        prompt = f"""You are a query rewrite system. Generate useful rewrites for the following query.

Domain Context:
{json.dumps(domain_knowledge, indent=2)}

Original Query: "{query}"

Generate rewrites of the following types:
{instructions}

For each rewrite, provide:
- The rewritten query
- The type of rewrite
- Confidence score (0.0 to 1.0)
- Brief reasoning for why this rewrite is useful

Respond in JSON format:
{{
    "rewrites": [
        {{
            "rewrite": "rewritten query text",
            "rewrite_type": "synonym|broader|expansion|substitute",
            "confidence": 0.9,
            "reasoning": "why this rewrite is useful"
        }}
    ]
}}"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a query rewrite expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            rewrites = []
            
            for r in result.get("rewrites", []):
                rewrites.append(QueryRewrite(
                    rewrite=r.get("rewrite", ""),
                    rewrite_type=RewriteType(r.get("rewrite_type", "synonym")),
                    confidence=float(r.get("confidence", 0.5)),
                    reasoning=r.get("reasoning", "")
                ))
            
            # Cache results
            self.query_cache[cache_key] = [
                {
                    "rewrite": r.rewrite,
                    "rewrite_type": r.rewrite_type.value,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning
                }
                for r in rewrites
            ]
            
            return rewrites
            
        except Exception as e:
            return []
    
    def apply_guardrails(
        self, 
        query: str, 
        results: List[Dict], 
        min_relevance: float = 0.7
    ) -> Tuple[List[Dict], List[str]]:
        """
        Apply post-processing guardrails to filter results.
        Validates semantic relevance and filters out low-quality matches.
        """
        if not results:
            return [], []
        
        filtered_results = []
        filtered_reasons = []
        
        # Generate query embedding for semantic similarity check
        try:
            query_embedding_response = self.openai.embeddings.create(
                input=query,
                model="text-embedding-3-large",
                dimensions=1024
            )
            query_embedding = query_embedding_response.data[0].embedding
        except Exception as e:
            # If embedding fails, return all results with warning
            return results, [f"Guardrail check failed: {str(e)}"]
        
        # Check each result
        for result in results:
            score = result.get('score', 0.0)
            metadata = result.get('metadata', {})
            
            # Guardrail 1: Minimum relevance threshold
            if score < min_relevance:
                filtered_reasons.append(
                    f"Result {metadata.get('chunk_id', 'unknown')} filtered: "
                    f"relevance score {score:.2f} below threshold {min_relevance}"
                )
                continue
            
            # Guardrail 2: Check for required metadata fields
            required_fields = ['summary', 'timestamp']
            missing_fields = [f for f in required_fields if not metadata.get(f)]
            if missing_fields:
                filtered_reasons.append(
                    f"Result {metadata.get('chunk_id', 'unknown')} filtered: "
                    f"missing required fields: {', '.join(missing_fields)}"
                )
                continue
            
            # Guardrail 3: Semantic similarity check (if we have embeddings)
            # This would require storing embeddings with results, which we can add later
            
            filtered_results.append(result)
        
        return filtered_results, filtered_reasons
    
    def _get_relevant_domain_context(self, query: str, context: Optional[Dict] = None) -> Dict:
        """
        Retrieve relevant domain context for RAG.
        This simulates the context-engineering approach from Instacart.
        """
        domain_context = {
            "common_topics": self.domain_context["common_topics"],
            "query_patterns": self.domain_context["query_patterns"]
        }
        
        # Add any provided context
        if context:
            domain_context.update(context)
        
        # Determine relevant topics based on query
        relevant_topics = []
        query_lower = query.lower()
        for topic in self.domain_context["common_topics"]:
            if topic.lower() in query_lower:
                relevant_topics.append(topic)
        
        domain_context["relevant_topics"] = relevant_topics
        
        return domain_context
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100) if total_queries > 0 else 0
        
        return {
            "cache_size": len(self.query_cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1f}%"
        }
    
    def clear_cache(self):
        """Clear the query cache"""
        self.query_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

