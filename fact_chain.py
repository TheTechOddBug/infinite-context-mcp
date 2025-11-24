#!/usr/bin/env python3
"""
Fact Chain Module - Pre-computed Fact Chains for Memory System
Extracts atomic facts during ingestion and chains related facts together.
Enables fast retrieval without query-time graph traversal.
"""
import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class FactType(Enum):
    """Types of facts"""
    STATEMENT = "statement"  # General statement
    PREFERENCE = "preference"  # User preference
    DECISION = "decision"  # Decision made
    EVENT = "event"  # Something that happened
    RELATIONSHIP = "relationship"  # Relationship between entities
    STATE = "state"  # Current state of something
    CAPABILITY = "capability"  # What something can do
    REQUIREMENT = "requirement"  # What is needed
    PROBLEM = "problem"  # Issue or problem
    SOLUTION = "solution"  # Solution to a problem


@dataclass
class Fact:
    """An atomic fact extracted from content"""
    id: str
    content: str  # The fact statement
    fact_type: FactType
    confidence: float  # 0.0 to 1.0
    
    # Temporal validity
    valid_from: str
    valid_until: Optional[str] = None
    is_current: bool = True
    
    # Source tracking
    source_chunk_id: str = ""
    source_timestamp: str = ""
    
    # Entity references
    entities: List[str] = field(default_factory=list)  # Entity names mentioned
    
    # Chaining
    related_fact_ids: List[str] = field(default_factory=list)
    supersedes_fact_id: Optional[str] = None
    superseded_by_fact_id: Optional[str] = None
    
    # Clustering
    cluster_id: Optional[str] = None
    
    # User scoping
    user_id: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['fact_type'] = self.fact_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Fact':
        data['fact_type'] = FactType(data['fact_type'])
        return cls(**data)


@dataclass
class FactCluster:
    """A cluster of related facts"""
    id: str
    name: str  # Descriptive name
    fact_ids: List[str] = field(default_factory=list)
    summary: str = ""  # Pre-computed summary
    primary_entities: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FactCluster':
        return cls(**data)


class FactChainManager:
    """
    Manages fact extraction, chaining, and clustering.
    Pre-computes relationships during ingestion for fast retrieval.
    """
    
    FACT_PREFIX = "fact_"
    CLUSTER_PREFIX = "cluster_"
    
    def __init__(self, pinecone_index, openai_client: Optional[OpenAI] = None):
        self.index = pinecone_index
        self.openai = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # In-memory fact store
        self._facts: Dict[str, Fact] = {}
        self._clusters: Dict[str, FactCluster] = {}
        
        # Index structures for fast lookup
        self._entity_to_facts: Dict[str, Set[str]] = defaultdict(set)
        self._type_to_facts: Dict[FactType, Set[str]] = defaultdict(set)
        self._cluster_to_facts: Dict[str, Set[str]] = defaultdict(set)
        
        # User context
        self._user_id: Optional[str] = None
    
    def set_user(self, user_id: str):
        """Set current user context"""
        if self._user_id != user_id:
            self._user_id = user_id
            self._facts.clear()
            self._clusters.clear()
            self._entity_to_facts.clear()
            self._type_to_facts.clear()
            self._cluster_to_facts.clear()
    
    def _generate_fact_id(self, content: str) -> str:
        """Generate unique fact ID"""
        combined = f"{self._user_id or 'global'}:{content[:100]}"
        return f"{self.FACT_PREFIX}{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def _generate_cluster_id(self, name: str) -> str:
        """Generate unique cluster ID"""
        combined = f"{self._user_id or 'global'}:{name}"
        return f"{self.CLUSTER_PREFIX}{hashlib.sha256(combined.encode()).hexdigest()[:12]}"
    
    async def extract_facts(
        self, 
        content: str,
        source_chunk_id: str,
        existing_entities: Optional[List[str]] = None
    ) -> List[Fact]:
        """Extract atomic facts from content using LLM"""
        prompt = f"""Extract atomic facts from the following content. Each fact should be:
- A single, self-contained statement
- Specific and verifiable
- Not redundant with other facts

Content:
{content[:5000]}

For each fact, provide:
- content: The fact statement (1-2 sentences)
- type: One of: statement, preference, decision, event, relationship, state, capability, requirement, problem, solution
- confidence: How certain is this fact (0.0-1.0)
- entities: List of entity names mentioned in this fact
- is_temporal: Whether this fact has a time limit (true/false)

Respond in JSON format:
{{
    "facts": [
        {{
            "content": "The project uses Python 3.11",
            "type": "statement",
            "confidence": 0.95,
            "entities": ["Python"],
            "is_temporal": false
        }},
        {{
            "content": "User prefers dark mode interfaces",
            "type": "preference",
            "confidence": 0.8,
            "entities": [],
            "is_temporal": false
        }}
    ]
}}

Focus on:
- Important decisions and their rationale
- User preferences and patterns
- Technical requirements and constraints
- Problems encountered and solutions found
- Relationships between concepts/people/projects"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting atomic facts from text. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            now = datetime.now().isoformat()
            
            facts = []
            for f_data in result.get("facts", []):
                fact_content = f_data.get("content", "")
                if not fact_content:
                    continue
                
                try:
                    fact_type = FactType(f_data.get("type", "statement"))
                except ValueError:
                    fact_type = FactType.STATEMENT
                
                fact = Fact(
                    id=self._generate_fact_id(fact_content),
                    content=fact_content,
                    fact_type=fact_type,
                    confidence=float(f_data.get("confidence", 0.7)),
                    valid_from=now,
                    valid_until=None if not f_data.get("is_temporal") else None,
                    is_current=True,
                    source_chunk_id=source_chunk_id,
                    source_timestamp=now,
                    entities=f_data.get("entities", []),
                    user_id=self._user_id or ""
                )
                
                # Check for supersession
                await self._check_supersession(fact)
                
                # Store fact
                self._facts[fact.id] = fact
                
                # Update indexes
                for entity in fact.entities:
                    self._entity_to_facts[entity.lower()].add(fact.id)
                self._type_to_facts[fact.fact_type].add(fact.id)
                
                facts.append(fact)
            
            # Chain related facts
            await self._chain_facts(facts)
            
            # Update clusters
            await self._update_clusters(facts)
            
            return facts
            
        except Exception as e:
            logger.warning(f"Error extracting facts: {e}")
            return []
    
    async def _check_supersession(self, new_fact: Fact):
        """Check if new fact supersedes existing facts"""
        # Find facts with similar entities and type
        candidate_facts = []
        
        for entity in new_fact.entities:
            entity_lower = entity.lower()
            if entity_lower in self._entity_to_facts:
                for fact_id in self._entity_to_facts[entity_lower]:
                    old_fact = self._facts.get(fact_id)
                    if old_fact and old_fact.fact_type == new_fact.fact_type:
                        candidate_facts.append(old_fact)
        
        if not candidate_facts:
            return
        
        # Use LLM to check for supersession
        prompt = f"""Does the new fact supersede (replace/update) any of the existing facts?

New fact: "{new_fact.content}"

Existing facts:
{chr(10).join([f'{i+1}. "{f.content}"' for i, f in enumerate(candidate_facts[:5])])}

Respond in JSON:
{{
    "supersedes": null or 1-5 (the number of the fact being superseded),
    "reasoning": "brief explanation"
}}

A fact supersedes another if:
- It updates the same information with newer data
- It contradicts the previous fact
- It provides a more current state of the same thing"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at detecting fact supersession. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            supersedes_idx = result.get("supersedes")
            
            if supersedes_idx and 1 <= supersedes_idx <= len(candidate_facts):
                old_fact = candidate_facts[supersedes_idx - 1]
                
                # Mark old fact as superseded
                old_fact.is_current = False
                old_fact.superseded_by_fact_id = new_fact.id
                
                # Link new fact to old
                new_fact.supersedes_fact_id = old_fact.id
                
                logger.info(f"Fact supersession: '{new_fact.content[:50]}...' supersedes '{old_fact.content[:50]}...'")
                
        except Exception as e:
            logger.warning(f"Error checking supersession: {e}")
    
    async def _chain_facts(self, facts: List[Fact]):
        """Chain related facts together"""
        if len(facts) < 2:
            return
        
        # Find relationships between facts based on shared entities
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts):
                if i >= j:
                    continue
                
                # Check for shared entities
                shared_entities = set(e.lower() for e in fact1.entities) & set(e.lower() for e in fact2.entities)
                
                if shared_entities:
                    # Link facts
                    if fact2.id not in fact1.related_fact_ids:
                        fact1.related_fact_ids.append(fact2.id)
                    if fact1.id not in fact2.related_fact_ids:
                        fact2.related_fact_ids.append(fact1.id)
        
        # Also chain with existing facts
        for fact in facts:
            for entity in fact.entities:
                entity_lower = entity.lower()
                for existing_fact_id in self._entity_to_facts.get(entity_lower, set()):
                    if existing_fact_id != fact.id:
                        existing_fact = self._facts.get(existing_fact_id)
                        if existing_fact and existing_fact_id not in fact.related_fact_ids:
                            fact.related_fact_ids.append(existing_fact_id)
                            if fact.id not in existing_fact.related_fact_ids:
                                existing_fact.related_fact_ids.append(fact.id)
    
    async def _update_clusters(self, new_facts: List[Fact]):
        """Update fact clusters with new facts"""
        # Group facts by primary entity
        entity_facts: Dict[str, List[Fact]] = defaultdict(list)
        
        for fact in new_facts:
            if fact.entities:
                primary_entity = fact.entities[0].lower()
                entity_facts[primary_entity].append(fact)
        
        # Create or update clusters
        for entity, facts in entity_facts.items():
            cluster_id = self._generate_cluster_id(entity)
            
            if cluster_id in self._clusters:
                # Update existing cluster
                cluster = self._clusters[cluster_id]
                for fact in facts:
                    if fact.id not in cluster.fact_ids:
                        cluster.fact_ids.append(fact.id)
                        fact.cluster_id = cluster_id
                cluster.updated_at = datetime.now().isoformat()
            else:
                # Create new cluster
                now = datetime.now().isoformat()
                cluster = FactCluster(
                    id=cluster_id,
                    name=entity.title(),
                    fact_ids=[f.id for f in facts],
                    primary_entities=[entity],
                    created_at=now,
                    updated_at=now
                )
                self._clusters[cluster_id] = cluster
                
                for fact in facts:
                    fact.cluster_id = cluster_id
            
            # Update cluster index
            for fact in facts:
                self._cluster_to_facts[cluster_id].add(fact.id)
        
        # Generate cluster summaries periodically
        for cluster_id in entity_facts.keys():
            cluster_id_full = self._generate_cluster_id(cluster_id)
            cluster = self._clusters.get(cluster_id_full)
            if cluster and len(cluster.fact_ids) >= 3 and not cluster.summary:
                cluster.summary = await self._generate_cluster_summary(cluster)
    
    async def _generate_cluster_summary(self, cluster: FactCluster) -> str:
        """Generate a summary for a fact cluster"""
        facts = [self._facts.get(fid) for fid in cluster.fact_ids[:10]]
        facts = [f for f in facts if f]
        
        if not facts:
            return ""
        
        facts_text = "\n".join([f"- {f.content}" for f in facts])
        
        prompt = f"""Summarize these related facts about "{cluster.name}" in 1-2 sentences:

{facts_text}

Provide a concise summary that captures the key information."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=100
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Error generating cluster summary: {e}")
            return ""
    
    def get_facts_by_entity(self, entity: str, current_only: bool = True) -> List[Fact]:
        """Get facts mentioning an entity"""
        entity_lower = entity.lower()
        fact_ids = self._entity_to_facts.get(entity_lower, set())
        
        facts = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact:
                if current_only and not fact.is_current:
                    continue
                facts.append(fact)
        
        return sorted(facts, key=lambda f: f.source_timestamp, reverse=True)
    
    def get_facts_by_type(self, fact_type: FactType, current_only: bool = True) -> List[Fact]:
        """Get facts of a specific type"""
        fact_ids = self._type_to_facts.get(fact_type, set())
        
        facts = []
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact:
                if current_only and not fact.is_current:
                    continue
                facts.append(fact)
        
        return sorted(facts, key=lambda f: f.source_timestamp, reverse=True)
    
    def get_related_facts(self, fact_id: str, max_depth: int = 2) -> List[Fact]:
        """Get facts related to a given fact through chains"""
        if fact_id not in self._facts:
            return []
        
        visited = {fact_id}
        result = []
        queue = [(fact_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            current_fact = self._facts.get(current_id)
            if not current_fact:
                continue
            
            for related_id in current_fact.related_fact_ids:
                if related_id not in visited:
                    visited.add(related_id)
                    related_fact = self._facts.get(related_id)
                    if related_fact:
                        result.append(related_fact)
                        queue.append((related_id, depth + 1))
        
        return result
    
    def get_cluster_facts(self, cluster_id: str) -> List[Fact]:
        """Get all facts in a cluster"""
        cluster = self._clusters.get(cluster_id)
        if not cluster:
            return []
        
        return [self._facts.get(fid) for fid in cluster.fact_ids if self._facts.get(fid)]
    
    def get_current_facts(self, limit: int = 50) -> List[Fact]:
        """Get current (non-superseded) facts"""
        current = [f for f in self._facts.values() if f.is_current]
        return sorted(current, key=lambda f: f.source_timestamp, reverse=True)[:limit]
    
    def get_fact_context(self, query: str, max_facts: int = 10) -> str:
        """Get relevant fact context for a query"""
        # Simple keyword matching for now
        query_words = set(query.lower().split())
        
        scored_facts = []
        for fact in self._facts.values():
            if not fact.is_current:
                continue
            
            # Score by keyword overlap
            fact_words = set(fact.content.lower().split())
            entity_words = set(e.lower() for e in fact.entities)
            
            overlap = len(query_words & (fact_words | entity_words))
            if overlap > 0:
                scored_facts.append((fact, overlap))
        
        # Sort by score
        scored_facts.sort(key=lambda x: x[1], reverse=True)
        
        # Build context string
        context_parts = []
        for fact, score in scored_facts[:max_facts]:
            context_parts.append(f"- {fact.content}")
        
        return "\n".join(context_parts)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of fact store"""
        type_counts = defaultdict(int)
        for fact in self._facts.values():
            type_counts[fact.fact_type.value] += 1
        
        current_count = sum(1 for f in self._facts.values() if f.is_current)
        superseded_count = len(self._facts) - current_count
        
        return {
            "total_facts": len(self._facts),
            "current_facts": current_count,
            "superseded_facts": superseded_count,
            "clusters": len(self._clusters),
            "fact_types": dict(type_counts),
            "entities_tracked": len(self._entity_to_facts)
        }
    
    async def save_facts(self):
        """Save facts to Pinecone"""
        if not self._user_id:
            return
        
        facts_id = f"facts_{self._user_id}"
        
        # Generate embedding for fact content
        all_facts_text = "\n".join([f.content for f in list(self._facts.values())[:100]])
        
        response = self.openai.embeddings.create(
            input=all_facts_text[:8000] if all_facts_text else "empty facts",
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding
        
        # Serialize data
        facts_data = {
            "facts": {k: v.to_dict() for k, v in self._facts.items()},
            "clusters": {k: v.to_dict() for k, v in self._clusters.items()}
        }
        
        self.index.upsert([{
            "id": facts_id,
            "values": embedding,
            "metadata": {
                "type": "fact_store",
                "user_id": self._user_id,
                "facts_data": json.dumps(facts_data)[:35000],
                "updated_at": datetime.now().isoformat(),
                "fact_count": len(self._facts),
                "cluster_count": len(self._clusters)
            }
        }])
        
        logger.info(f"Saved {len(self._facts)} facts for user {self._user_id}")
    
    async def load_facts(self, user_id: str) -> bool:
        """Load facts from Pinecone"""
        self.set_user(user_id)
        facts_id = f"facts_{user_id}"
        
        try:
            result = self.index.fetch(ids=[facts_id])
            
            if result and result.get('vectors') and facts_id in result['vectors']:
                metadata = result['vectors'][facts_id].get('metadata', {})
                facts_data = json.loads(metadata.get('facts_data', '{}'))
                
                # Load facts
                for k, v in facts_data.get("facts", {}).items():
                    fact = Fact.from_dict(v)
                    self._facts[k] = fact
                    
                    # Rebuild indexes
                    for entity in fact.entities:
                        self._entity_to_facts[entity.lower()].add(k)
                    self._type_to_facts[fact.fact_type].add(k)
                    if fact.cluster_id:
                        self._cluster_to_facts[fact.cluster_id].add(k)
                
                # Load clusters
                for k, v in facts_data.get("clusters", {}).items():
                    self._clusters[k] = FactCluster.from_dict(v)
                
                logger.info(f"Loaded {len(self._facts)} facts for user {user_id}")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error loading facts for {user_id}: {e}")
            return False


