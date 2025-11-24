#!/usr/bin/env python3
"""
User Profile Module - Stateful Memory System
Manages user profiles that inject default context into every conversation.
Inspired by Supermemory's approach to user understanding.
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import logging

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class EntityInfo:
    """Information about an entity (person, project, concept, etc.)"""
    name: str
    entity_type: str  # person, project, concept, location, organization
    first_seen: str
    last_seen: str
    mention_count: int = 1
    related_entities: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EntityInfo':
        return cls(**data)


@dataclass
class UserPreference:
    """A user preference or behavioral pattern"""
    category: str  # communication_style, topic_interest, work_pattern, etc.
    key: str
    value: Any
    confidence: float  # 0.0 to 1.0
    last_updated: str
    source_count: int = 1  # Number of contexts supporting this preference
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserPreference':
        return cls(**data)


@dataclass
class TemporalContext:
    """Current temporal context for the user"""
    current_focus: Optional[str] = None
    recent_topics: List[str] = field(default_factory=list)
    active_projects: List[str] = field(default_factory=list)
    last_interaction: Optional[str] = None
    session_start: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TemporalContext':
        return cls(**data)


@dataclass
class UserProfile:
    """Complete user profile for stateful memory"""
    user_id: str
    created_at: str
    updated_at: str
    
    # Core profile data
    preferences: Dict[str, UserPreference] = field(default_factory=dict)
    entities: Dict[str, EntityInfo] = field(default_factory=dict)
    
    # Behavioral patterns
    topic_interests: Dict[str, float] = field(default_factory=dict)  # topic -> interest score
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)
    
    # Temporal context
    temporal_context: TemporalContext = field(default_factory=TemporalContext)
    
    # Profile summary (pre-computed for injection)
    profile_summary: str = ""
    
    # Statistics
    total_contexts: int = 0
    total_facts: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "preferences": {k: v.to_dict() for k, v in self.preferences.items()},
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "topic_interests": self.topic_interests,
            "interaction_patterns": self.interaction_patterns,
            "temporal_context": self.temporal_context.to_dict(),
            "profile_summary": self.profile_summary,
            "total_contexts": self.total_contexts,
            "total_facts": self.total_facts
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProfile':
        preferences = {k: UserPreference.from_dict(v) for k, v in data.get("preferences", {}).items()}
        entities = {k: EntityInfo.from_dict(v) for k, v in data.get("entities", {}).items()}
        temporal_context = TemporalContext.from_dict(data.get("temporal_context", {}))
        
        return cls(
            user_id=data["user_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            preferences=preferences,
            entities=entities,
            topic_interests=data.get("topic_interests", {}),
            interaction_patterns=data.get("interaction_patterns", {}),
            temporal_context=temporal_context,
            profile_summary=data.get("profile_summary", ""),
            total_contexts=data.get("total_contexts", 0),
            total_facts=data.get("total_facts", 0)
        )


class UserProfileManager:
    """
    Manages user profiles with stateful memory.
    Builds and maintains user understanding across conversations.
    """
    
    # Profile namespace prefix for Pinecone
    PROFILE_PREFIX = "profile_"
    
    def __init__(self, pinecone_index, openai_client: Optional[OpenAI] = None):
        self.index = pinecone_index
        self.openai = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # In-memory cache for active profiles
        self._profile_cache: Dict[str, UserProfile] = {}
        self._cache_ttl = timedelta(hours=1)
        self._cache_timestamps: Dict[str, datetime] = {}
        
        # Default user ID for single-user mode
        self.default_user_id = os.getenv("DEFAULT_USER_ID", "default_user")
    
    def _generate_profile_id(self, user_id: str) -> str:
        """Generate a unique profile ID for storage"""
        return f"{self.PROFILE_PREFIX}{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"
    
    def get_or_create_profile(self, user_id: Optional[str] = None) -> UserProfile:
        """Get existing profile or create a new one"""
        user_id = user_id or self.default_user_id
        
        # Check cache first
        if user_id in self._profile_cache:
            cache_time = self._cache_timestamps.get(user_id)
            if cache_time and datetime.now() - cache_time < self._cache_ttl:
                return self._profile_cache[user_id]
        
        # Try to load from Pinecone
        profile = self._load_profile_from_storage(user_id)
        
        if profile is None:
            # Create new profile
            now = datetime.now().isoformat()
            profile = UserProfile(
                user_id=user_id,
                created_at=now,
                updated_at=now
            )
            logger.info(f"Created new profile for user: {user_id}")
        
        # Update cache
        self._profile_cache[user_id] = profile
        self._cache_timestamps[user_id] = datetime.now()
        
        return profile
    
    def _load_profile_from_storage(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from Pinecone"""
        try:
            profile_id = self._generate_profile_id(user_id)
            
            # Fetch from Pinecone
            result = self.index.fetch(ids=[profile_id])
            
            if result and result.get('vectors') and profile_id in result['vectors']:
                metadata = result['vectors'][profile_id].get('metadata', {})
                profile_data = json.loads(metadata.get('profile_data', '{}'))
                
                if profile_data:
                    return UserProfile.from_dict(profile_data)
            
            return None
        except Exception as e:
            logger.warning(f"Error loading profile for {user_id}: {e}")
            return None
    
    async def save_profile(self, profile: UserProfile):
        """Save user profile to Pinecone"""
        try:
            profile.updated_at = datetime.now().isoformat()
            profile_id = self._generate_profile_id(profile.user_id)
            
            # Generate embedding for profile summary
            embedding = await self._generate_profile_embedding(profile)
            
            # Store in Pinecone
            self.index.upsert([{
                "id": profile_id,
                "values": embedding,
                "metadata": {
                    "type": "user_profile",
                    "user_id": profile.user_id,
                    "profile_data": json.dumps(profile.to_dict()),
                    "updated_at": profile.updated_at,
                    "profile_summary": profile.profile_summary[:2000]
                }
            }])
            
            # Update cache
            self._profile_cache[profile.user_id] = profile
            self._cache_timestamps[profile.user_id] = datetime.now()
            
            logger.info(f"Saved profile for user: {profile.user_id}")
        except Exception as e:
            logger.error(f"Error saving profile for {profile.user_id}: {e}")
            raise
    
    async def _generate_profile_embedding(self, profile: UserProfile) -> List[float]:
        """Generate embedding for user profile"""
        # Create a text representation of the profile
        profile_text = self._profile_to_text(profile)
        
        response = self.openai.embeddings.create(
            input=profile_text[:8000],
            model="text-embedding-3-large",
            dimensions=1024
        )
        return response.data[0].embedding
    
    def _profile_to_text(self, profile: UserProfile) -> str:
        """Convert profile to text for embedding"""
        parts = [f"User Profile for {profile.user_id}"]
        
        # Add top interests
        if profile.topic_interests:
            top_interests = sorted(profile.topic_interests.items(), key=lambda x: x[1], reverse=True)[:10]
            parts.append(f"Top interests: {', '.join([t[0] for t in top_interests])}")
        
        # Add key entities
        if profile.entities:
            people = [e.name for e in profile.entities.values() if e.entity_type == 'person'][:5]
            projects = [e.name for e in profile.entities.values() if e.entity_type == 'project'][:5]
            if people:
                parts.append(f"Key people: {', '.join(people)}")
            if projects:
                parts.append(f"Active projects: {', '.join(projects)}")
        
        # Add preferences
        if profile.preferences:
            pref_strs = [f"{p.key}: {p.value}" for p in list(profile.preferences.values())[:10]]
            parts.append(f"Preferences: {'; '.join(pref_strs)}")
        
        # Add temporal context
        if profile.temporal_context.current_focus:
            parts.append(f"Current focus: {profile.temporal_context.current_focus}")
        if profile.temporal_context.recent_topics:
            parts.append(f"Recent topics: {', '.join(profile.temporal_context.recent_topics[:5])}")
        
        return "\n".join(parts)
    
    async def update_profile_from_context(
        self, 
        profile: UserProfile,
        content: str,
        summary: str,
        topics: List[str],
        extracted_entities: Optional[Dict[str, List[str]]] = None,
        extracted_facts: Optional[List[Dict]] = None
    ) -> UserProfile:
        """
        Update user profile based on new context.
        This builds cumulative understanding of the user.
        """
        now = datetime.now().isoformat()
        
        # Update topic interests
        for topic in topics:
            topic_lower = topic.lower()
            current_score = profile.topic_interests.get(topic_lower, 0.0)
            # Decay existing score and add new contribution
            profile.topic_interests[topic_lower] = min(1.0, current_score * 0.9 + 0.1)
        
        # Update entities if provided
        if extracted_entities:
            for entity_type, entity_names in extracted_entities.items():
                for name in entity_names:
                    entity_key = f"{entity_type}:{name.lower()}"
                    if entity_key in profile.entities:
                        entity = profile.entities[entity_key]
                        entity.last_seen = now
                        entity.mention_count += 1
                    else:
                        profile.entities[entity_key] = EntityInfo(
                            name=name,
                            entity_type=entity_type,
                            first_seen=now,
                            last_seen=now
                        )
        
        # Update temporal context
        profile.temporal_context.last_interaction = now
        profile.temporal_context.recent_topics = topics[:10]
        
        # Infer current focus from topics
        if topics:
            profile.temporal_context.current_focus = topics[0]
        
        # Update statistics
        profile.total_contexts += 1
        if extracted_facts:
            profile.total_facts += len(extracted_facts)
        
        # Regenerate profile summary
        profile.profile_summary = await self._generate_profile_summary(profile)
        
        return profile
    
    async def _generate_profile_summary(self, profile: UserProfile) -> str:
        """Generate a concise profile summary for context injection"""
        prompt = f"""Generate a concise profile summary for context injection. This will be used to personalize AI responses.

User Profile Data:
- Total contexts: {profile.total_contexts}
- Total facts: {profile.total_facts}
- Top interests: {list(sorted(profile.topic_interests.items(), key=lambda x: x[1], reverse=True)[:5])}
- Key entities: {[f"{e.name} ({e.entity_type})" for e in list(profile.entities.values())[:10]]}
- Recent topics: {profile.temporal_context.recent_topics[:5]}
- Current focus: {profile.temporal_context.current_focus}

Generate a 2-3 sentence summary that captures:
1. What the user is primarily interested in
2. Key people/projects they work with
3. Their current focus

Keep it factual and concise. This will be injected as context."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a profile summarization expert. Generate concise, factual summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"Error generating profile summary: {e}")
            return f"User interested in: {', '.join(profile.temporal_context.recent_topics[:3])}"
    
    async def extract_entities_from_content(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from content using LLM"""
        prompt = f"""Extract entities from the following content. Categorize them as:
- people: Names of people mentioned
- projects: Project or product names
- concepts: Technical concepts, technologies, or methodologies
- organizations: Company or organization names
- locations: Places mentioned

Content:
{content[:4000]}

Respond in JSON format:
{{
    "people": ["name1", "name2"],
    "projects": ["project1"],
    "concepts": ["concept1", "concept2"],
    "organizations": ["org1"],
    "locations": ["location1"]
}}

Only include entities that are clearly mentioned. Omit empty categories."""

        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an entity extraction expert. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            # Filter out empty lists
            return {k: v for k, v in result.items() if v}
        except Exception as e:
            logger.warning(f"Error extracting entities: {e}")
            return {}
    
    def get_profile_context(self, profile: UserProfile) -> str:
        """
        Get the profile context to inject into queries.
        This is the key feature - default context without searching.
        """
        if not profile.profile_summary:
            return ""
        
        context_parts = [
            "=== User Context ===",
            profile.profile_summary
        ]
        
        # Add current focus if available
        if profile.temporal_context.current_focus:
            context_parts.append(f"\nCurrent focus: {profile.temporal_context.current_focus}")
        
        # Add recent topics
        if profile.temporal_context.recent_topics:
            context_parts.append(f"Recent topics: {', '.join(profile.temporal_context.recent_topics[:5])}")
        
        # Add key entities
        active_entities = sorted(
            profile.entities.values(),
            key=lambda e: e.mention_count,
            reverse=True
        )[:5]
        if active_entities:
            entity_strs = [f"{e.name} ({e.entity_type})" for e in active_entities]
            context_parts.append(f"Key entities: {', '.join(entity_strs)}")
        
        context_parts.append("===================")
        
        return "\n".join(context_parts)
    
    def apply_forgetfulness(self, profile: UserProfile, decay_factor: float = 0.95) -> UserProfile:
        """
        Apply forgetfulness mechanism to deprioritize old/unused information.
        This keeps profiles relevant and prevents context pollution.
        """
        now = datetime.now()
        
        # Decay topic interests
        for topic in list(profile.topic_interests.keys()):
            profile.topic_interests[topic] *= decay_factor
            # Remove topics with very low scores
            if profile.topic_interests[topic] < 0.01:
                del profile.topic_interests[topic]
        
        # Decay entity relevance based on last seen
        entities_to_remove = []
        for key, entity in profile.entities.items():
            try:
                last_seen = datetime.fromisoformat(entity.last_seen)
                days_since_seen = (now - last_seen).days
                
                # Remove entities not seen in 90 days with low mention count
                if days_since_seen > 90 and entity.mention_count < 3:
                    entities_to_remove.append(key)
            except:
                pass
        
        for key in entities_to_remove:
            del profile.entities[key]
        
        return profile
    
    def clear_cache(self):
        """Clear the profile cache"""
        self._profile_cache.clear()
        self._cache_timestamps.clear()


