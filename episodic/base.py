"""
Base Context Store interface and abstract classes.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from .core import Context, ContextFilter, ContextUpdate


class BaseContextStore(ABC):
    """
    Abstract base class for Context Store implementations.
    Defines the interface that all context store implementations must follow.
    """
    
    def __init__(self, endpoint: str = "", api_key: str = "", namespace: str = "default"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.default_namespace = namespace
    
    @abstractmethod
    async def store(self, context_id: str, data: Dict[str, Any], text: str = None, 
                   ttl: int = None, tags: List[str] = None, namespace: str = None,
                   context_type: str = "generic", metadata: Dict[str, Any] = None) -> Context:
        """
        Store a context with the given data.
        
        Args:
            context_id: Unique identifier for the context
            data: Structured data to store
            text: Optional text representation
            ttl: Time to live in seconds
            tags: List of tags for categorization
            namespace: Namespace for the context (uses default if not provided)
            context_type: Type of context
            metadata: Additional metadata
            
        Returns:
            The stored Context object
        """
        pass
    
    @abstractmethod
    async def store_context(self, context: Context) -> Context:
        """
        Store a Context object directly.
        
        Args:
            context: Context object to store
            
        Returns:
            The stored Context object
        """
        pass
    
    @abstractmethod
    async def get(self, context_id: str) -> Context:
        """
        Get a specific context by ID.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            The Context object
            
        Raises:
            ContextNotFoundException: If context is not found or expired
        """
        pass
    
    @abstractmethod
    async def query(self, filter: ContextFilter) -> List[Context]:
        """
        Query contexts based on the provided filter.
        
        Args:
            filter: ContextFilter object specifying query criteria
            
        Returns:
            List of matching Context objects
        """
        pass
    
    @abstractmethod
    async def search_text(self, query: str, namespaces: List[str] = None, 
                         limit: int = 10) -> List[Context]:
        """
        Search contexts by text content.
        
        Args:
            query: Text query string
            namespaces: Optional list of namespaces to search in
            limit: Maximum number of results to return
            
        Returns:
            List of matching Context objects
        """
        pass
    
    @abstractmethod
    async def search_semantic(self, query: str, namespaces: List[str] = None,
                             similarity_threshold: float = 0.7, limit: int = 10) -> List[Context]:
        """
        Search contexts using semantic similarity (vector embeddings).
        
        Args:
            query: Text query string
            namespaces: Optional list of namespaces to search in
            similarity_threshold: Minimum similarity score (0.0-1.0)
            limit: Maximum number of results to return
            
        Returns:
            List of matching Context objects with similarity scores
        """
        pass
    
    @abstractmethod
    async def search_hybrid(self, text_query: str, filters: ContextFilter, 
                           limit: int = 15) -> List[Context]:
        """
        Hybrid search combining text search and metadata filters.
        
        Args:
            text_query: Text query string
            filters: ContextFilter for metadata filtering
            limit: Maximum number of results to return
            
        Returns:
            List of matching Context objects
        """
        pass
    
    @abstractmethod
    async def delete(self, context_id: str) -> bool:
        """
        Delete a context by ID.
        
        Args:
            context_id: ID of the context to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    async def compose(self, composition_id: str, components: List[Dict[str, str]], 
                     merge_strategy: str = "priority_weighted") -> Context:
        """
        Compose multiple contexts into a single context.
        
        Args:
            composition_id: ID for the composed context
            components: List of component specifications
            merge_strategy: Strategy for merging contexts
            
        Returns:
            The composed Context object
        """
        pass
    
    @abstractmethod
    def add_subscriber(self, subscriber):
        """
        Add a subscriber for context updates.
        
        Args:
            subscriber: Subscriber object with handle_context_update method
        """
        pass
    
    @abstractmethod
    def remove_subscriber(self, subscriber):
        """
        Remove a subscriber.
        
        Args:
            subscriber: Subscriber object to remove
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Return health status of the context store.
        
        Returns:
            Dictionary containing health status information
        """
        pass
    
    @abstractmethod
    async def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return diagnostic information.
        
        Returns:
            Dictionary containing diagnostic information
        """
        pass 
