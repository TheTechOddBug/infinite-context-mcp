#!/usr/bin/env python3
"""
Entity Graph Module - Knowledge Graph for Memory System
Builds and traverses relationship graphs between entities.
Enables multi-hop reasoning and relationship-aware search.
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


class RelationType(Enum):
    """Types of relationships between entities"""
    # Professional relationships
    WORKS_WITH = "works_with"
    WORKS_AT = "works_at"
    MANAGES = "manages"
    REPORTS_TO = "reports_to"
    
    # Project relationships
    WORKS_ON = "works_on"
    CREATED = "created"
    OWNS = "owns"
    CONTRIBUTES_TO = "contributes_to"
    
    # Conceptual relationships
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    USES = "uses"
    
    # Social relationships
    KNOWS = "knows"
    INTRODUCED_BY = "introduced_by"
    MET_AT = "met_at"
    
    # Causal relationships
    CAUSED_BY = "caused_by"
    RESULTS_IN = "results_in"
    SUPERSEDES = "supersedes"
    
    # Generic
    ASSOCIATED_WITH = "associated_with"


@dataclass
class Entity:
    """An entity in the knowledge graph"""
    id: str
    name: str
    entity_type: str  # person, project, concept, organization, location, event
    attributes: Dict[str, Any] = field(default_factory=dict)
    first_seen: str = ""
    last_seen: str = ""
    mention_count: int = 1
    source_chunks: List[str] = field(default_factory=list)  # Chunk IDs where mentioned
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(**data)


@dataclass
class Relationship:
    """A relationship between two entities"""
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    strength: float = 1.0  # 0.0 to 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    source_chunks: List[str] = field(default_factory=list)  # Chunk IDs where relationship found
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['relation_type'] = self.relation_type.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Relationship':
        data['relation_type'] = RelationType(data['relation_type'])
        return cls(**data)


@dataclass
class GraphTraversalResult:
    """Result of a graph traversal query"""
    path: List[str]  # Entity IDs in path
    entities: List[Entity]
    relationships: List[Relationship]
    total_strength: float
    explanation: str


class EntityGraph:
    """
    Knowledge graph for entity relationships.
    Enables multi-hop reasoning and relationship-aware search.
    """
    
    # Graph namespace prefix for Pinecone
    GRAPH_PREFIX = "graph_"
    ENTITY_PREFIX = "entity_"
    RELATION_PREFIX = "relation_"
    
    def __init__(self, pinecone_index, openai_client: Optional[OpenAI] = None):
        self.index = pinecone_index
        self.openai = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # In-memory graph cache for fast traversal
        self._entities: Dict[str, Entity] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._adjacency: Dict[str, List[str]] = defaultdict(list)  # entity_id -> [relationship_ids]
        
        # User-scoped graphs
        self._user_id: Optional[str] = None
    
    def set_user(self, user_id: str):
        """Set the current user for graph operations"""
        if self._user_id != user_id:
            self._user_id = user_id
            # Clear cache when user changes
            self._entities.clear()
            self._relationships.clear()
            self._adjacency.clear()
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate unique entity ID"""
        combined = f"{self._user_id or 'global'}:{entity_type}:{name.lower()}"
        return f"{self.ENTITY_PREFIX}{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    def _generate_relationship_id(self, source_id: str, target_id: str, relation_type: RelationType) -> str:
        """Generate unique relationship ID"""
        combined = f"{source_id}:{relation_type.value}:{target_id}"
        return f"{self.RELATION_PREFIX}{hashlib.sha256(combined.encode()).hexdigest()[:16]}"
    
    async def add_entity(
        self, 
        name: str, 
        entity_type: str, 
        attributes: Optional[Dict] = None,
        source_chunk_id: Optional[str] = None
    ) -> Entity:
        """Add or update an entity in the graph"""
        entity_id = self._generate_entity_id(name, entity_type)
        now = datetime.now().isoformat()
        
        if entity_id in self._entities:
            # Update existing entity
            entity = self._entities[entity_id]
            entity.last_seen = now
            entity.mention_count += 1
            if attributes:
                entity.attributes.update(attributes)
            if source_chunk_id and source_chunk_id not in entity.source_chunks:
                entity.source_chunks.append(source_chunk_id)
        else:
            # Create new entity
            entity = Entity(
                id=entity_id,
                name=name,
                entity_type=entity_type,
                attributes=attributes or {},
                first_seen=now,
                last_seen=now,
                source_chunks=[source_chunk_id] if source_chunk_id else []
            )
            self._entities[entity_id] = entity
        
        return entity
    
    async def add_relationship(
        self,
        source_entity: Entity,
        target_entity: Entity,
        relation_type: RelationType,
        strength: float = 1.0,
        attributes: Optional[Dict] = None,
        source_chunk_id: Optional[str] = None
    ) -> Relationship:
        """Add or strengthen a relationship between entities"""
        rel_id = self._generate_relationship_id(
            source_entity.id, target_entity.id, relation_type
        )
        now = datetime.now().isoformat()
        
        if rel_id in self._relationships:
            # Strengthen existing relationship
            rel = self._relationships[rel_id]
            rel.strength = min(1.0, rel.strength + 0.1)  # Increase strength
            if attributes:
                rel.attributes.update(attributes)
            if source_chunk_id and source_chunk_id not in rel.source_chunks:
                rel.source_chunks.append(source_chunk_id)
        else:
            # Create new relationship
            rel = Relationship(
                id=rel_id,
                source_entity_id=source_entity.id,
                target_entity_id=target_entity.id,
                relation_type=relation_type,
                strength=strength,
                attributes=attributes or {},
                created_at=now,
                source_chunks=[source_chunk_id] if source_chunk_id else []
            )
            self._relationships[rel_id] = rel
            
            # Update adjacency list (bidirectional)
            self._adjacency[source_entity.id].append(rel_id)
            self._adjacency[target_entity.id].append(rel_id)
        
        return rel
    
    async def extract_entities_and_relationships(
        self, 
        content: str,
        source_chunk_id: Optional[str] = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from content using LLM"""
        prompt = f"""Analyze the following content and extract:
1. Entities (people, projects, concepts, organizations, locations, events)
2. Relationships between entities

Content:
{content[:6000]}

For each entity, provide:
- name: The entity name
- type: person, project, concept, organization, location, or event
- attributes: Any relevant attributes (optional)

For each relationship, provide:
- source: Source entity name
- target: Target entity name
- type: One of: works_with, works_at, manages, reports_to, works_on, created, owns, contributes_to, related_to, part_of, depends_on, uses, knows, introduced_by, met_at, caused_by, results_in, supersedes, associated_with
- attributes: Any relevant attributes (optional)

Respond in JSON format:
{{
    "entities": [
        {{"name": "...", "type": "person", "attributes": {{}}}},
        ...
    ],
    "relationships": [
        {{"source": "...", "target": "...", "type": "works_with", "attributes": {{}}}},
        ...
    ]
}}

Focus on:
- Named entities (specific people, projects, technologies)
- Clear relationships between them
- Avoid generic or vague entities"""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at extracting entities and relationships from text. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Process entities
            entities = []
            entity_map = {}  # name -> Entity for relationship linking
            
            for e_data in result.get("entities", []):
                entity = await self.add_entity(
                    name=e_data.get("name", ""),
                    entity_type=e_data.get("type", "concept"),
                    attributes=e_data.get("attributes", {}),
                    source_chunk_id=source_chunk_id
                )
                entities.append(entity)
                entity_map[e_data.get("name", "").lower()] = entity
            
            # Process relationships
            relationships = []
            for r_data in result.get("relationships", []):
                source_name = r_data.get("source", "").lower()
                target_name = r_data.get("target", "").lower()
                
                if source_name in entity_map and target_name in entity_map:
                    try:
                        rel_type = RelationType(r_data.get("type", "associated_with"))
                    except ValueError:
                        rel_type = RelationType.ASSOCIATED_WITH
                    
                    rel = await self.add_relationship(
                        source_entity=entity_map[source_name],
                        target_entity=entity_map[target_name],
                        relation_type=rel_type,
                        attributes=r_data.get("attributes", {}),
                        source_chunk_id=source_chunk_id
                    )
                    relationships.append(rel)
            
            return entities, relationships
            
        except Exception as e:
            logger.warning(f"Error extracting entities and relationships: {e}")
            return [], []
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> Optional[Entity]:
        """Get entity by name"""
        for entity in self._entities.values():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    return entity
        return None
    
    def get_related_entities(
        self, 
        entity_id: str, 
        relation_types: Optional[List[RelationType]] = None,
        max_depth: int = 1
    ) -> List[Tuple[Entity, Relationship]]:
        """Get entities related to the given entity"""
        if entity_id not in self._entities:
            return []
        
        results = []
        visited = {entity_id}
        queue = [(entity_id, 0)]  # (entity_id, depth)
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if depth >= max_depth:
                continue
            
            # Get relationships for this entity
            for rel_id in self._adjacency.get(current_id, []):
                rel = self._relationships.get(rel_id)
                if not rel:
                    continue
                
                # Filter by relation type if specified
                if relation_types and rel.relation_type not in relation_types:
                    continue
                
                # Get the other entity in the relationship
                other_id = rel.target_entity_id if rel.source_entity_id == current_id else rel.source_entity_id
                
                if other_id not in visited:
                    visited.add(other_id)
                    other_entity = self._entities.get(other_id)
                    if other_entity:
                        results.append((other_entity, rel))
                        queue.append((other_id, depth + 1))
        
        return results
    
    def traverse_path(
        self, 
        start_entity_id: str, 
        end_entity_id: str,
        max_depth: int = 3
    ) -> Optional[GraphTraversalResult]:
        """Find path between two entities using BFS"""
        if start_entity_id not in self._entities or end_entity_id not in self._entities:
            return None
        
        # BFS to find shortest path
        visited = {start_entity_id}
        queue = [(start_entity_id, [start_entity_id], [], 1.0)]  # (current, path, rels, strength)
        
        while queue:
            current_id, path, rels, strength = queue.pop(0)
            
            if current_id == end_entity_id:
                # Found path
                entities = [self._entities[eid] for eid in path]
                relationships = [self._relationships[rid] for rid in rels]
                
                return GraphTraversalResult(
                    path=path,
                    entities=entities,
                    relationships=relationships,
                    total_strength=strength,
                    explanation=self._generate_path_explanation(entities, relationships)
                )
            
            if len(path) >= max_depth:
                continue
            
            # Explore neighbors
            for rel_id in self._adjacency.get(current_id, []):
                rel = self._relationships.get(rel_id)
                if not rel:
                    continue
                
                other_id = rel.target_entity_id if rel.source_entity_id == current_id else rel.source_entity_id
                
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((
                        other_id,
                        path + [other_id],
                        rels + [rel_id],
                        strength * rel.strength
                    ))
        
        return None
    
    def _generate_path_explanation(self, entities: List[Entity], relationships: List[Relationship]) -> str:
        """Generate human-readable explanation of a path"""
        if not entities:
            return ""
        
        parts = [entities[0].name]
        for i, rel in enumerate(relationships):
            parts.append(f"--[{rel.relation_type.value}]-->")
            parts.append(entities[i + 1].name)
        
        return " ".join(parts)
    
    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Find all entities of a given type"""
        return [e for e in self._entities.values() if e.entity_type == entity_type]
    
    def get_entity_context(self, entity_id: str, max_related: int = 5) -> str:
        """Get contextual information about an entity for injection"""
        entity = self._entities.get(entity_id)
        if not entity:
            return ""
        
        context_parts = [f"{entity.name} ({entity.entity_type})"]
        
        # Add attributes
        if entity.attributes:
            attrs = [f"{k}: {v}" for k, v in list(entity.attributes.items())[:3]]
            context_parts.append(f"  Attributes: {', '.join(attrs)}")
        
        # Add related entities
        related = self.get_related_entities(entity_id, max_depth=1)[:max_related]
        if related:
            rel_strs = [f"{e.name} ({r.relation_type.value})" for e, r in related]
            context_parts.append(f"  Related: {', '.join(rel_strs)}")
        
        return "\n".join(context_parts)
    
    def get_graph_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the graph"""
        entity_types = defaultdict(int)
        relation_types = defaultdict(int)
        
        for entity in self._entities.values():
            entity_types[entity.entity_type] += 1
        
        for rel in self._relationships.values():
            relation_types[rel.relation_type.value] += 1
        
        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "entity_types": dict(entity_types),
            "relation_types": dict(relation_types),
            "most_connected": self._get_most_connected_entities(5)
        }
    
    def _get_most_connected_entities(self, n: int) -> List[Dict]:
        """Get the n most connected entities"""
        connections = []
        for entity_id, rel_ids in self._adjacency.items():
            entity = self._entities.get(entity_id)
            if entity:
                connections.append({
                    "name": entity.name,
                    "type": entity.entity_type,
                    "connections": len(rel_ids)
                })
        
        connections.sort(key=lambda x: x["connections"], reverse=True)
        return connections[:n]
    
    async def save_graph(self):
        """Save the graph to Pinecone"""
        if not self._user_id:
            logger.warning("No user set, cannot save graph")
            return
        
        graph_id = f"{self.GRAPH_PREFIX}{self._user_id}"
        
        # Generate embedding for graph content
        graph_text = self._graph_to_text()
        
        response = self.openai.embeddings.create(
            input=graph_text[:8000],
            model="text-embedding-3-large",
            dimensions=1024
        )
        embedding = response.data[0].embedding
        
        # Serialize graph data
        graph_data = {
            "entities": {k: v.to_dict() for k, v in self._entities.items()},
            "relationships": {k: v.to_dict() for k, v in self._relationships.items()}
        }
        
        # Store in Pinecone
        self.index.upsert([{
            "id": graph_id,
            "values": embedding,
            "metadata": {
                "type": "entity_graph",
                "user_id": self._user_id,
                "graph_data": json.dumps(graph_data)[:35000],  # Pinecone limit
                "updated_at": datetime.now().isoformat(),
                "entity_count": len(self._entities),
                "relationship_count": len(self._relationships)
            }
        }])
        
        logger.info(f"Saved graph for user {self._user_id}: {len(self._entities)} entities, {len(self._relationships)} relationships")
    
    async def load_graph(self, user_id: str) -> bool:
        """Load graph from Pinecone"""
        self.set_user(user_id)
        graph_id = f"{self.GRAPH_PREFIX}{user_id}"
        
        try:
            result = self.index.fetch(ids=[graph_id])
            
            if result and result.get('vectors') and graph_id in result['vectors']:
                metadata = result['vectors'][graph_id].get('metadata', {})
                graph_data = json.loads(metadata.get('graph_data', '{}'))
                
                # Load entities
                for k, v in graph_data.get("entities", {}).items():
                    self._entities[k] = Entity.from_dict(v)
                
                # Load relationships
                for k, v in graph_data.get("relationships", {}).items():
                    self._relationships[k] = Relationship.from_dict(v)
                    # Rebuild adjacency list
                    rel = self._relationships[k]
                    self._adjacency[rel.source_entity_id].append(k)
                    self._adjacency[rel.target_entity_id].append(k)
                
                logger.info(f"Loaded graph for user {user_id}: {len(self._entities)} entities, {len(self._relationships)} relationships")
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error loading graph for {user_id}: {e}")
            return False
    
    def _graph_to_text(self) -> str:
        """Convert graph to text for embedding"""
        parts = []
        
        # Add entities by type
        by_type = defaultdict(list)
        for entity in self._entities.values():
            by_type[entity.entity_type].append(entity.name)
        
        for entity_type, names in by_type.items():
            parts.append(f"{entity_type.title()}: {', '.join(names[:20])}")
        
        # Add key relationships
        for rel in list(self._relationships.values())[:50]:
            source = self._entities.get(rel.source_entity_id)
            target = self._entities.get(rel.target_entity_id)
            if source and target:
                parts.append(f"{source.name} {rel.relation_type.value} {target.name}")
        
        return "\n".join(parts)
    
    def apply_decay(self, decay_factor: float = 0.95):
        """Apply decay to relationship strengths"""
        for rel in self._relationships.values():
            rel.strength *= decay_factor
        
        # Remove very weak relationships
        to_remove = [rid for rid, rel in self._relationships.items() if rel.strength < 0.1]
        for rid in to_remove:
            rel = self._relationships.pop(rid)
            # Clean up adjacency
            if rel.source_entity_id in self._adjacency:
                self._adjacency[rel.source_entity_id] = [
                    r for r in self._adjacency[rel.source_entity_id] if r != rid
                ]
            if rel.target_entity_id in self._adjacency:
                self._adjacency[rel.target_entity_id] = [
                    r for r in self._adjacency[rel.target_entity_id] if r != rid
                ]


