#!/usr/bin/env python3
"""
Memory Scorer Module - Hybrid Scoring for Memory System
Combines semantic, relational, temporal, and profile signals for ranking.
Implements the core scoring algorithm that makes memory better than RAG.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class ScoreComponent(Enum):
    """Components of the hybrid score"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    PROFILE = "profile"
    TEMPORAL = "temporal"
    ENTITY = "entity"
    FACT = "fact"
    RECENCY = "recency"
    RELATIONSHIP = "relationship"


@dataclass
class ScoringWeights:
    """Weights for different scoring components"""
    semantic: float = 0.35  # Base vector similarity
    keyword: float = 0.15  # Keyword matching
    profile: float = 0.15  # User profile relevance
    temporal: float = 0.10  # Temporal validity
    entity: float = 0.10  # Entity overlap
    fact: float = 0.05  # Fact relevance
    recency: float = 0.05  # How recent
    relationship: float = 0.05  # Relationship graph signals
    
    def normalize(self):
        """Ensure weights sum to 1.0"""
        total = (self.semantic + self.keyword + self.profile + self.temporal + 
                 self.entity + self.fact + self.recency + self.relationship)
        if total > 0:
            self.semantic /= total
            self.keyword /= total
            self.profile /= total
            self.temporal /= total
            self.entity /= total
            self.fact /= total
            self.recency /= total
            self.relationship /= total


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of a result's score"""
    total_score: float
    components: Dict[str, float]
    explanations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "total_score": self.total_score,
            "components": self.components,
            "explanations": self.explanations
        }


class MemoryScorer:
    """
    Advanced hybrid scorer for memory system results.
    Combines multiple signals to rank results beyond simple cosine similarity.
    """
    
    def __init__(
        self, 
        weights: Optional[ScoringWeights] = None,
        openai_client: Optional[OpenAI] = None
    ):
        self.weights = weights or ScoringWeights()
        self.weights.normalize()
        self.openai = openai_client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def score_result(
        self,
        result: Dict,
        query: str,
        profile_context: Optional[Dict] = None,
        temporal_context: Optional[Dict] = None,
        entity_context: Optional[Dict] = None,
        fact_context: Optional[Dict] = None
    ) -> ScoreBreakdown:
        """
        Calculate comprehensive score for a search result.
        Combines all memory system signals.
        """
        components = {}
        explanations = []
        
        meta = result.get('metadata', {})
        
        # 1. Semantic Score (base vector similarity)
        semantic_score = result.get('score', 0.0)
        components[ScoreComponent.SEMANTIC.value] = semantic_score
        
        # 2. Keyword Score
        keyword_score, keyword_explanation = self._calculate_keyword_score(query, meta)
        components[ScoreComponent.KEYWORD.value] = keyword_score
        if keyword_explanation:
            explanations.append(keyword_explanation)
        
        # 3. Profile Score
        profile_score, profile_explanation = self._calculate_profile_score(meta, profile_context)
        components[ScoreComponent.PROFILE.value] = profile_score
        if profile_explanation:
            explanations.append(profile_explanation)
        
        # 4. Temporal Score
        temporal_score, temporal_explanation = self._calculate_temporal_score(meta, temporal_context)
        components[ScoreComponent.TEMPORAL.value] = temporal_score
        if temporal_explanation:
            explanations.append(temporal_explanation)
        
        # 5. Entity Score
        entity_score, entity_explanation = self._calculate_entity_score(query, meta, entity_context)
        components[ScoreComponent.ENTITY.value] = entity_score
        if entity_explanation:
            explanations.append(entity_explanation)
        
        # 6. Fact Score
        fact_score, fact_explanation = self._calculate_fact_score(query, meta, fact_context)
        components[ScoreComponent.FACT.value] = fact_score
        if fact_explanation:
            explanations.append(fact_explanation)
        
        # 7. Recency Score
        recency_score, recency_explanation = self._calculate_recency_score(meta)
        components[ScoreComponent.RECENCY.value] = recency_score
        if recency_explanation:
            explanations.append(recency_explanation)
        
        # 8. Relationship Score
        relationship_score, rel_explanation = self._calculate_relationship_score(meta, entity_context)
        components[ScoreComponent.RELATIONSHIP.value] = relationship_score
        if rel_explanation:
            explanations.append(rel_explanation)
        
        # Calculate weighted total
        total_score = (
            self.weights.semantic * components[ScoreComponent.SEMANTIC.value] +
            self.weights.keyword * components[ScoreComponent.KEYWORD.value] +
            self.weights.profile * components[ScoreComponent.PROFILE.value] +
            self.weights.temporal * components[ScoreComponent.TEMPORAL.value] +
            self.weights.entity * components[ScoreComponent.ENTITY.value] +
            self.weights.fact * components[ScoreComponent.FACT.value] +
            self.weights.recency * components[ScoreComponent.RECENCY.value] +
            self.weights.relationship * components[ScoreComponent.RELATIONSHIP.value]
        )
        
        return ScoreBreakdown(
            total_score=total_score,
            components=components,
            explanations=explanations
        )
    
    def _calculate_keyword_score(self, query: str, meta: Dict) -> Tuple[float, str]:
        """Calculate keyword matching score"""
        query_words = set(query.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                      'by', 'from', 'as', 'into', 'through', 'during', 'before',
                      'after', 'above', 'below', 'between', 'under', 'again',
                      'further', 'then', 'once', 'here', 'there', 'when', 'where',
                      'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                      'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
                      'or', 'because', 'until', 'while', 'about', 'against',
                      'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                      'those', 'am', 'it', 'its', 'i', 'me', 'my', 'myself',
                      'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours'}
        
        query_words = query_words - stop_words
        
        if not query_words:
            return 0.5, ""
        
        # Build content text
        content_text = ' '.join([
            str(meta.get('summary', '')),
            str(meta.get('content', '')),
            str(meta.get('topics', '')),
            str(meta.get('findings', ''))
        ]).lower()
        
        # Count matches
        matches = sum(1 for word in query_words if word in content_text)
        score = matches / len(query_words)
        
        # Exact phrase bonus
        if query.lower() in content_text:
            score = min(1.0, score + 0.3)
        
        explanation = f"Keyword match: {matches}/{len(query_words)} words" if matches > 0 else ""
        
        return score, explanation
    
    def _calculate_profile_score(self, meta: Dict, profile_context: Optional[Dict]) -> Tuple[float, str]:
        """Calculate profile relevance score"""
        if not profile_context:
            return 0.5, ""
        
        score = 0.0
        explanations = []
        
        # Check topic overlap
        profile_topics = set(t.lower() for t in profile_context.get('recent_topics', []))
        result_topics_str = str(meta.get('topics', '')).lower()
        
        if profile_topics:
            topic_matches = sum(1 for t in profile_topics if t in result_topics_str)
            if topic_matches > 0:
                score += 0.4 * (topic_matches / len(profile_topics))
                explanations.append(f"{topic_matches} topic matches")
        
        # Check entity overlap
        profile_entities = set(e.lower() for e in profile_context.get('entities', []))
        result_entities_str = str(meta.get('entities', '')).lower()
        
        if profile_entities:
            entity_matches = sum(1 for e in profile_entities if e in result_entities_str)
            if entity_matches > 0:
                score += 0.4 * (entity_matches / len(profile_entities))
                explanations.append(f"{entity_matches} entity matches")
        
        # Check current focus match
        current_focus = profile_context.get('current_focus', '').lower()
        if current_focus:
            content_text = str(meta.get('content', '')).lower() + str(meta.get('summary', '')).lower()
            if current_focus in content_text:
                score += 0.2
                explanations.append("Matches current focus")
        
        explanation = "; ".join(explanations) if explanations else ""
        return min(1.0, score), explanation
    
    def _calculate_temporal_score(self, meta: Dict, temporal_context: Optional[Dict]) -> Tuple[float, str]:
        """Calculate temporal validity score"""
        score = 1.0  # Default to fully valid
        explanation = ""
        
        now = datetime.now()
        
        # Check valid_from and valid_until
        valid_from_str = meta.get('valid_from', '')
        valid_until_str = meta.get('valid_until', '')
        
        if valid_from_str:
            try:
                valid_from = datetime.fromisoformat(valid_from_str.replace('Z', '+00:00'))
                if valid_from > now:
                    score = 0.3  # Not yet valid
                    explanation = "Not yet valid"
            except:
                pass
        
        if valid_until_str:
            try:
                valid_until = datetime.fromisoformat(valid_until_str.replace('Z', '+00:00'))
                if valid_until < now:
                    score = 0.2  # Expired
                    explanation = "Expired fact"
            except:
                pass
        
        # Apply temporal context from query
        if temporal_context:
            scope = temporal_context.get('scope', 'all_time')
            
            if scope == 'current' and score < 1.0:
                # For current queries, penalize non-current facts more
                score *= 0.5
            elif scope == 'historical':
                # For historical queries, don't penalize expired facts
                score = max(score, 0.7)
        
        return score, explanation
    
    def _calculate_entity_score(self, query: str, meta: Dict, entity_context: Optional[Dict]) -> Tuple[float, str]:
        """Calculate entity overlap score"""
        score = 0.5  # Neutral default
        explanation = ""
        
        # Extract entities from query (simple approach)
        query_words = set(w for w in query.split() if w[0].isupper() if len(w) > 1)
        
        if not query_words and not entity_context:
            return score, ""
        
        # Get result entities
        result_entities_str = str(meta.get('entities', ''))
        
        # Check query entity matches
        if query_words:
            matches = sum(1 for w in query_words if w.lower() in result_entities_str.lower())
            if matches > 0:
                score = 0.5 + 0.5 * (matches / len(query_words))
                explanation = f"{matches} entity matches from query"
        
        # Check entity context matches
        if entity_context and entity_context.get('query_entities'):
            context_entities = entity_context['query_entities']
            matches = sum(1 for e in context_entities if e.lower() in result_entities_str.lower())
            if matches > 0:
                score = max(score, 0.5 + 0.5 * (matches / len(context_entities)))
                if not explanation:
                    explanation = f"{matches} entity matches from context"
        
        return score, explanation
    
    def _calculate_fact_score(self, query: str, meta: Dict, fact_context: Optional[Dict]) -> Tuple[float, str]:
        """Calculate fact relevance score"""
        if not fact_context:
            return 0.5, ""
        
        score = 0.5
        explanation = ""
        
        # Check if this result contains relevant facts
        relevant_facts = fact_context.get('relevant_facts', [])
        chunk_id = meta.get('chunk_id', '')
        
        fact_matches = sum(1 for f in relevant_facts if f.get('source_chunk_id') == chunk_id)
        
        if fact_matches > 0:
            score = min(1.0, 0.5 + 0.1 * fact_matches)
            explanation = f"{fact_matches} relevant facts"
        
        return score, explanation
    
    def _calculate_recency_score(self, meta: Dict) -> Tuple[float, str]:
        """Calculate recency score"""
        timestamp_str = meta.get('timestamp', '')
        
        if not timestamp_str:
            return 0.5, ""
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            now = datetime.now()
            days_old = (now - timestamp).days
            
            # Exponential decay
            # Full score for last 7 days, then decay
            if days_old <= 7:
                score = 1.0
            elif days_old <= 30:
                score = 0.9 - 0.02 * (days_old - 7)
            elif days_old <= 90:
                score = 0.7 - 0.005 * (days_old - 30)
            else:
                score = max(0.3, 0.5 - 0.001 * (days_old - 90))
            
            explanation = f"{days_old} days old"
            return score, explanation
            
        except:
            return 0.5, ""
    
    def _calculate_relationship_score(self, meta: Dict, entity_context: Optional[Dict]) -> Tuple[float, str]:
        """Calculate relationship graph score"""
        if not entity_context:
            return 0.5, ""
        
        score = 0.5
        explanation = ""
        
        # Check if result entities are in the relationship graph
        related_entities = entity_context.get('related_entities', [])
        result_entities_str = str(meta.get('entities', '')).lower()
        
        if related_entities:
            matches = sum(1 for e in related_entities if e.lower() in result_entities_str)
            if matches > 0:
                score = min(1.0, 0.5 + 0.1 * matches)
                explanation = f"{matches} related entity matches"
        
        return score, explanation
    
    def rank_results(
        self,
        results: List[Dict],
        query: str,
        profile_context: Optional[Dict] = None,
        temporal_context: Optional[Dict] = None,
        entity_context: Optional[Dict] = None,
        fact_context: Optional[Dict] = None
    ) -> List[Tuple[Dict, ScoreBreakdown]]:
        """
        Rank all results using hybrid scoring.
        Returns results with their score breakdowns.
        """
        scored_results = []
        
        for result in results:
            breakdown = self.score_result(
                result=result,
                query=query,
                profile_context=profile_context,
                temporal_context=temporal_context,
                entity_context=entity_context,
                fact_context=fact_context
            )
            
            # Add score to result
            result['hybrid_score'] = breakdown.total_score
            result['score_breakdown'] = breakdown.to_dict()
            
            scored_results.append((result, breakdown))
        
        # Sort by total score
        scored_results.sort(key=lambda x: x[1].total_score, reverse=True)
        
        return scored_results
    
    def adjust_weights_for_query_type(self, query_type: str) -> ScoringWeights:
        """Adjust weights based on query type"""
        weights = ScoringWeights()
        
        if query_type == 'question':
            # Questions benefit more from semantic matching
            weights.semantic = 0.45
            weights.keyword = 0.15
            weights.fact = 0.15
            weights.profile = 0.10
            weights.temporal = 0.05
            weights.entity = 0.05
            weights.recency = 0.03
            weights.relationship = 0.02
            
        elif query_type == 'temporal':
            # Temporal queries emphasize time-based signals
            weights.temporal = 0.30
            weights.semantic = 0.25
            weights.recency = 0.15
            weights.keyword = 0.10
            weights.profile = 0.10
            weights.entity = 0.05
            weights.fact = 0.03
            weights.relationship = 0.02
            
        elif query_type == 'context_retrieval':
            # Context retrieval benefits from profile and relationship signals
            weights.profile = 0.25
            weights.relationship = 0.20
            weights.semantic = 0.20
            weights.entity = 0.15
            weights.keyword = 0.10
            weights.temporal = 0.05
            weights.recency = 0.03
            weights.fact = 0.02
            
        elif query_type == 'search':
            # Search benefits from keyword and semantic
            weights.semantic = 0.35
            weights.keyword = 0.25
            weights.entity = 0.15
            weights.profile = 0.10
            weights.temporal = 0.05
            weights.recency = 0.05
            weights.fact = 0.03
            weights.relationship = 0.02
        
        weights.normalize()
        return weights
    
    def explain_score(self, breakdown: ScoreBreakdown) -> str:
        """Generate human-readable score explanation"""
        parts = [f"Total Score: {breakdown.total_score:.2%}"]
        
        # Show top contributing components
        sorted_components = sorted(
            breakdown.components.items(),
            key=lambda x: x[1],
            reverse=True
        )[:4]
        
        component_strs = []
        for comp, score in sorted_components:
            weight = getattr(self.weights, comp, 0)
            contribution = weight * score
            component_strs.append(f"{comp}: {score:.2f} (contrib: {contribution:.2%})")
        
        parts.append("Components: " + ", ".join(component_strs))
        
        if breakdown.explanations:
            parts.append("Details: " + "; ".join(breakdown.explanations[:3]))
        
        return " | ".join(parts)


