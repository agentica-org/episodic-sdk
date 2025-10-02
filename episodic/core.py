"""
Core data structures for the Context Store implementation.
"""

import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from fnmatch import fnmatch


@dataclass
class Context:
    """
    Represents a context object with structured data and text representation.
    """
    id: str
    data: Dict[str, Any]
    text: Optional[str] = None
    namespace: str = "default"
    type: str = "generic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    ttl: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    auto_render_text: bool = False
    
    # Enhanced fields for semantic search
    embedding: Optional[List[float]] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_generated_at: Optional[float] = None
    
    # Search result scores
    semantic_similarity: Optional[float] = None
    text_rank: Optional[float] = None
    hybrid_score: Optional[float] = None
    
    def __post_init__(self):
        """Set expiration time if TTL is provided."""
        if self.ttl:
            self.expires_at = self.created_at + self.ttl
        
        # Auto-render text if requested and text is not provided
        if self.auto_render_text and not self.text:
            self.text = self._generate_text()
    
    def _generate_text(self) -> str:
        """Generate a text representation from structured data."""
        if not self.data:
            return f"Context {self.id}"
        
        # Simple text generation based on data structure
        text_parts = []
        for key, value in self.data.items():
            if isinstance(value, (int, float, str, bool)):
                text_parts.append(f"{key}: {value}")
        
        return f"{self.type} - {', '.join(text_parts)}"
    
    def is_expired(self) -> bool:
        """Check if the context has expired."""
        return self.expires_at is not None and time.time() > self.expires_at
    
    def matches_tags(self, tags: List[str]) -> bool:
        """Check if context has any of the specified tags."""
        return bool(set(tags) & set(self.tags))
    
    def matches_namespace(self, namespace_pattern: str) -> bool:
        """Check if context namespace matches a pattern (supports wildcards)."""
        return fnmatch(self.namespace, namespace_pattern)
    
    def has_embedding(self) -> bool:
        """
        Check if context has a valid embedding.
        
        Returns:
            True if embedding exists and is valid
        """
        if not self.embedding:
            return False
        
        # Check if embedding has expected dimensions (384 for default model)
        expected_dims = 384  # Could be made configurable
        if len(self.embedding) != expected_dims:
            return False
        
        # Check if embedding is not all zeros
        return not all(abs(x) < 1e-10 for x in self.embedding)
    
    def is_embedding_stale(self, max_age_hours: int = 24) -> bool:
        """
        Check if embedding is stale based on age.
        
        Args:
            max_age_hours: Maximum age in hours before considering stale
            
        Returns:
            True if embedding is stale or missing
        """
        if not self.embedding_generated_at:
            return True
        
        max_age_seconds = max_age_hours * 3600
        current_time = time.time()
        
        return (current_time - self.embedding_generated_at) > max_age_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Context to dictionary for serialization."""
        return {
            "id": self.id,
            "data": self.data,
            "text": self.text,
            "namespace": self.namespace,
            "type": self.type,
            "metadata": self.metadata,
            "tags": self.tags,
            "ttl": self.ttl,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "auto_render_text": self.auto_render_text,
            "embedding": self.embedding,
            "embedding_model": self.embedding_model,
            "embedding_generated_at": self.embedding_generated_at,
            "semantic_similarity": self.semantic_similarity,
            "text_rank": self.text_rank,
            "hybrid_score": self.hybrid_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Context':
        """Create Context from dictionary."""
        return cls(
            id=data["id"],
            data=data["data"],
            text=data.get("text"),
            namespace=data.get("namespace", "default"),
            type=data.get("type", "generic"),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            ttl=data.get("ttl"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            expires_at=data.get("expires_at"),
            auto_render_text=data.get("auto_render_text", False),
            embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_generated_at=data.get("embedding_generated_at"),
            semantic_similarity=data.get("semantic_similarity"),
            text_rank=data.get("text_rank"),
            hybrid_score=data.get("hybrid_score")
        )


@dataclass
class ContextFilter:
    """
    Filter for querying contexts.
    """
    namespaces: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    context_types: List[str] = field(default_factory=list)
    since: Optional[str] = None  # "1h", "30m", "1d", etc.
    limit: int = 100
    include_expired: bool = False
    
    def matches(self, context: Context) -> bool:
        """Check if a context matches this filter."""
        # Check expiration
        if not self.include_expired and context.is_expired():
            return False
        
        # Check namespaces
        if self.namespaces:
            if not any(context.matches_namespace(ns) for ns in self.namespaces):
                return False
        
        # Check tags
        if self.tags:
            if not context.matches_tags(self.tags):
                return False
        
        # Check types
        if self.context_types:
            if context.type not in self.context_types:
                return False
        
        # Check time filter
        if self.since:
            since_timestamp = self._parse_time_filter(self.since)
            if context.created_at < since_timestamp:
                return False
        
        return True
    
    def _parse_time_filter(self, since_str: str) -> float:
        """Parse time filter string to timestamp."""
        current_time = time.time()
        
        if since_str.endswith('h'):
            hours = int(since_str[:-1])
            return current_time - (hours * 3600)
        elif since_str.endswith('m'):
            minutes = int(since_str[:-1])
            return current_time - (minutes * 60)
        elif since_str.endswith('d'):
            days = int(since_str[:-1])
            return current_time - (days * 86400)
        else:
            return current_time
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextFilter':
        """Create ContextFilter from dictionary."""
        return cls(
            namespaces=data.get("namespaces", []),
            tags=data.get("tags", []),
            context_types=data.get("context_types", []),
            since=data.get("since"),
            limit=data.get("limit", 100),
            include_expired=data.get("include_expired", False)
        )


@dataclass
class ContextUpdate:
    """
    Represents a context update event for subscriptions.
    """
    context: Context
    operation: str  # "create", "update", "delete"
    namespace: str
    timestamp: float = field(default_factory=time.time)


class ContextStoreException(Exception):
    """Base exception for Context Store operations."""
    pass


class ContextNotFoundException(ContextStoreException):
    """Exception raised when a context is not found."""
    pass


class SubscriptionException(ContextStoreException):
    """Exception raised for subscription-related errors."""
    pass