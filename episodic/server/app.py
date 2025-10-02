"""
FastAPI server implementation for Episodic Context Store.
Exposes the SqliteContextStore backend via HTTP REST API with WebSocket support.
Compatible with episodic-cloud API interface.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Query, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, HttpUrl

from ..store import SqliteContextStore
from ..core import Context, ContextFilter, ContextUpdate, ContextNotFoundException

logger = logging.getLogger(__name__)


# Enums for search functionality
class TextSearchMode(str, Enum):
    """Text search modes."""
    EXACT = "exact"
    PHRASE = "phrase"
    FUZZY = "fuzzy"

class SearchType(str, Enum):
    """Search result types."""
    TEXT = "text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class SearchStrategy(str, Enum):
    """Search strategies for hybrid search."""
    SEMANTIC_FIRST = "semantic_first"
    TEXT_FIRST = "text_first"
    BALANCED = "balanced"


# Pydantic models for API - Updated to match episodic-cloud interface
class ContextData(BaseModel):
    """Request model for storing context - matches episodic-cloud ContextData."""
    context_id: str = Field(..., alias="id")
    data: Dict[str, Any]
    text: Optional[str] = None
    namespace: str = "default"
    context_type: str = Field("generic", alias="type")
    metadata: Dict[str, Any] = {}
    tags: List[str] = []
    ttl: Optional[int] = None
    created_at: Optional[float] = None
    updated_at: Optional[float] = None
    expires_at: Optional[float] = None
    auto_render_text: bool = False

    class Config:
        populate_by_name = True


class StoreContextDirectRequest(BaseModel):
    """Request model for storing context object directly."""
    context: Dict[str, Any]  # Context object as dict


class ContextFilter(BaseModel):
    """Request model for querying contexts - matches episodic-cloud."""
    namespaces: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    context_types: Optional[List[str]] = None
    since: Optional[str] = None
    limit: int = 100
    include_expired: bool = False


class SearchRequest(BaseModel):
    """Basic search request model."""
    query: str
    namespaces: Optional[List[str]] = None
    limit: int = 10


class TextSearchRequest(BaseModel):
    """Request model for text search - matches episodic-cloud."""
    query: str
    namespaces: Optional[List[str]] = None
    search_mode: TextSearchMode = TextSearchMode.PHRASE
    include_ranking: bool = True
    rank_threshold: float = 0.0
    limit: int = 10

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('rank_threshold')
    def validate_rank_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Rank threshold must be between 0.0 and 1.0')
        return v

    @validator('limit')
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError('Limit must be between 1 and 100')
        return v


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search - matches episodic-cloud."""
    query: str
    namespaces: Optional[List[str]] = None
    similarity_threshold: float = 0.7
    include_similarity_score: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    limit: int = 10

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Similarity threshold must be between 0.0 and 1.0')
        return v

    @validator('limit')
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError('Limit must be between 1 and 100')
        return v


class HybridSemanticSearchRequest(BaseModel):
    """Request model for hybrid semantic search - matches episodic-cloud."""
    query: str
    filters: Optional[ContextFilter] = None
    search_strategy: SearchStrategy = SearchStrategy.BALANCED
    semantic_weight: float = 0.6
    text_weight: float = 0.4
    similarity_threshold: float = 0.5
    text_rank_threshold: float = 0.1
    limit: int = 15

    @validator('query')
    def validate_query(cls, v):
        if not v or not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @validator('semantic_weight', 'text_weight')
    def validate_weights(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Weights must be between 0.0 and 1.0')
        return v

    @validator('text_weight')
    def validate_weights_sum(cls, v, values):
        if 'semantic_weight' in values:
            weight_sum = v + values['semantic_weight']
            if not (0.99 <= weight_sum <= 1.01):  # 0.01 tolerance
                raise ValueError('Semantic weight + text weight must sum to 1.0')
        return v

    @validator('similarity_threshold', 'text_rank_threshold')
    def validate_thresholds(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('Thresholds must be between 0.0 and 1.0')
        return v

    @validator('limit')
    def validate_limit(cls, v):
        if not (1 <= v <= 100):
            raise ValueError('Limit must be between 1 and 100')
        return v


class HybridSearchRequest(BaseModel):
    """Request model for hybrid search."""
    text_query: str
    filters: ContextFilter
    limit: int = 15


class CompositionRequest(BaseModel):
    """Request model for context composition - matches episodic-cloud."""
    composition_id: str
    components: List[Dict[str, str]]
    merge_strategy: str = "priority_weighted"


class WebhookConfig(BaseModel):
    """Webhook configuration model."""
    url: HttpUrl
    secret: Optional[str] = None
    headers: Optional[Dict[str, str]] = {}


class Subscription(BaseModel):
    """Subscription model - matches episodic-cloud."""
    subscription_id: Optional[str] = None
    client_id: str
    delivery_method: str = "websocket"  # websocket or webhook
    webhook_config: Optional[WebhookConfig] = None
    filters: ContextFilter


class SearchResult(BaseModel):
    """Individual search result."""
    context: Dict[str, Any]
    semantic_similarity: Optional[float] = None
    text_rank: Optional[float] = None
    search_type: SearchType
    matched_fields: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response model for search operations."""
    results: List[SearchResult]
    total_found: int
    search_time_ms: float
    search_metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""
    texts: List[str]
    model: str = "all-MiniLM-L6-v2"

    @validator('texts')
    def validate_texts(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one text must be provided')
        return v


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    embeddings: List[List[float]]
    model: str
    dimensions: int
    generation_time_ms: float


class WebSocketManager:
    """Manages WebSocket connections and subscriptions."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_subscriptions: Dict[str, Set[str]] = {}  # client_id -> set of namespaces
        self.subscriptions: Dict[str, Dict] = {}  # subscription_id -> subscription data
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_subscriptions[client_id] = set()
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.client_subscriptions:
            del self.client_subscriptions[client_id]
        logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: str, client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast_update(self, update: ContextUpdate):
        """Broadcast context update to subscribed clients."""
        message = {
            "type": "context_update",
            "data": {
                "context": update.context.to_dict(),
                "operation": update.operation,
                "namespace": update.namespace,
                "timestamp": time.time()
            }
        }
        message_str = json.dumps(message)
        
        # Send to clients subscribed to this namespace
        for client_id, namespaces in self.client_subscriptions.items():
            if not namespaces or update.namespace in namespaces or "*" in namespaces:
                await self.send_personal_message(message_str, client_id)
    
    def subscribe_to_namespace(self, client_id: str, namespace: str):
        """Subscribe a client to a namespace."""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].add(namespace)
    
    def unsubscribe_from_namespace(self, client_id: str, namespace: str):
        """Unsubscribe a client from a namespace."""
        if client_id in self.client_subscriptions:
            self.client_subscriptions[client_id].discard(namespace)


class EpisodicServer:
    """Episodic Context Store Server."""
    
    def __init__(self, db_path: Optional[str] = None, namespace: str = "default"):
        """
        Initialize the Episodic server.
        
        Args:
            db_path: Path to SQLite database file
            namespace: Default namespace
        """
        self.context_store = SqliteContextStore(
            endpoint="sqlite://",
            namespace=namespace,
            db_path=db_path
        )
        self.websocket_manager = WebSocketManager()
        
        # Subscribe to context store updates to broadcast via WebSocket
        self.context_store.add_subscriber(self)
    
    async def handle_context_update(self, update: ContextUpdate):
        """Handle context updates from the store and broadcast to WebSocket clients."""
        await self.websocket_manager.broadcast_update(update)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Episodic Context Store Server")
    yield
    # Shutdown
    logger.info("Shutting down Episodic Context Store Server")
    if hasattr(app.state, 'server'):
        await app.state.server.context_store.close()


def create_app(db_path: Optional[str] = None, namespace: str = "default") -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        db_path: Path to SQLite database file
        namespace: Default namespace
        
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Episodic Context Store Server",
        description="HTTP API for Episodic Context Store with real-time WebSocket subscriptions",
        version="0.1.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize server
    server = EpisodicServer(db_path=db_path, namespace=namespace)
    app.state.server = server
    
    def get_server() -> EpisodicServer:
        """Dependency to get the server instance."""
        return app.state.server

    # API Key validation - matches episodic-cloud approach
    async def verify_api_key(x_api_key: Optional[str] = Header(None)):
        """Verify API key if configured."""
        # For local SQLite server, we don't enforce API keys by default
        # This can be configured if needed
        return True
    
    def _generate_text_from_data(data: Dict[str, Any], context_type: str, context_id: str) -> str:
        """Generate text representation from structured data."""
        if not data:
            return f"Context {context_id}"
        
        text_parts = []
        for key, value in data.items():
            if isinstance(value, (int, float, str, bool)):
                text_parts.append(f"{key}: {value}")
        
        return f"{context_type} - {', '.join(text_parts)}"

    def _context_to_dict(context: Context) -> Dict[str, Any]:
        """Convert Context object to dictionary matching episodic-cloud format."""
        context_dict = context.to_dict()
        # Ensure all expected fields are present
        context_dict.setdefault("auto_render_text", False)
        context_dict.setdefault("embedding", None)
        context_dict.setdefault("embedding_model", None)
        context_dict.setdefault("embedding_generated_at", None)
        return context_dict

    # Health check endpoint
    @app.get("/health")
    async def health_check(server: EpisodicServer = Depends(get_server)):
        """Health check endpoint."""
        try:
            health_result = await server.context_store.health_check()
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "services": {
                    "sqlite": "healthy",
                    "context_store": "healthy"
                },
                **health_result
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    @app.get("/diagnostics")
    async def get_diagnostics(server: EpisodicServer = Depends(get_server)):
        """Get diagnostic information."""
        try:
            diagnostics = await server.context_store.get_diagnostics()
            # Add WebSocket connection info
            diagnostics.update({
                "active_websocket_connections": len(server.websocket_manager.active_connections),
                "websocket_subscriptions": len(server.websocket_manager.subscriptions),
                "timestamp": datetime.utcnow().isoformat()
            })
            return diagnostics
        except Exception as e:
            return {"error": f"Diagnostics collection failed: {str(e)}"}
    
    @app.get("/metrics")
    async def metrics(server: EpisodicServer = Depends(get_server)):
        """Basic metrics endpoint - matches episodic-cloud interface."""
        try:
            diagnostics = await server.context_store.get_diagnostics()
            return {
                "active_contexts": diagnostics.get("total_contexts", 0),
                "active_websocket_connections": len(server.websocket_manager.active_connections),
                "websocket_subscriptions": len(server.websocket_manager.subscriptions),
                "webhook_subscriptions": 0,  # Not supported in local mode
                "webhook_deliveries_24h": {
                    "total": 0,
                    "successful": 0,
                    "failed": 0
                },
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Metrics error: {e}")
            return {"error": "Metrics collection failed"}
    
    @app.get("/auth/validate")
    async def auth_validate(_: bool = Depends(verify_api_key)):
        """Validate API key and return OK when valid."""
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Context management endpoints - Updated to match episodic-cloud interface
    @app.post("/contexts")
    async def store_context(
        context: ContextData,
        background_tasks: BackgroundTasks,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Store context data - main endpoint used by SDK."""
        try:
            current_time = time.time()
            context_id = context.context_id
            
            # Set timestamps
            if not context.created_at:
                context.created_at = current_time
            context.updated_at = current_time
            
            # Calculate expiration
            if context.ttl:
                context.expires_at = current_time + context.ttl
            
            # Generate text if auto_render_text is enabled and text is not provided
            text = context.text
            if context.auto_render_text and not text:
                text = _generate_text_from_data(context.data, context.context_type, context_id)
            
            stored_context = await server.context_store.store(
                context_id=context_id,
                data=context.data,
                text=text,
                ttl=context.ttl,
                tags=context.tags,
                namespace=context.namespace,
                context_type=context.context_type,
                metadata=context.metadata
            )
            
            return _context_to_dict(stored_context)
        except Exception as e:
            logger.error(f"Error storing context: {e}")
            raise HTTPException(status_code=500, detail="Storage failed")
    
    @app.post("/contexts/object")
    async def store_context_object(
        context: ContextData,
        background_tasks: BackgroundTasks,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Store a Context object directly - alternative endpoint."""
        return await store_context(context, background_tasks, server)
    
    @app.post("/contexts/direct")
    async def store_context_direct(
        request: StoreContextDirectRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Store a context object directly."""
        try:
            context = Context.from_dict(request.context)
            stored_context = await server.context_store.store_context(context)
            return _context_to_dict(stored_context)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/contexts/{context_id}")
    async def get_context(
        context_id: str,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Get a specific context by ID."""
        try:
            context = await server.context_store.get(context_id)
            return _context_to_dict(context)
        except ContextNotFoundException:
            raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/contexts/{context_id}")
    async def delete_context(
        context_id: str,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Delete a context by ID."""
        try:
            success = await server.context_store.delete(context_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Context '{context_id}' not found")
            return {"status": "deleted", "context_id": context_id}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/contexts/query")
    async def query_contexts(
        filter: ContextFilter,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Query contexts based on filters."""
        try:
            filter_obj = ContextFilter(
                namespaces=filter.namespaces,
                tags=filter.tags,
                context_types=filter.context_types,
                since=filter.since,
                limit=filter.limit
            )
            contexts = await server.context_store.query(filter_obj)
            return [_context_to_dict(context) for context in contexts]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/contexts/search/text", response_model=SearchResponse)
    async def search_text(
        request: TextSearchRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Search contexts by text content."""
        try:
            start_time = time.time()
            contexts = await server.context_store.search_text(
                query=request.query,
                namespaces=request.namespaces,
                limit=request.limit
            )
            search_time_ms = (time.time() - start_time) * 1000
            
            results = []
            for context in contexts:
                results.append(SearchResult(
                    context=_context_to_dict(context),
                    search_type=SearchType.TEXT,
                    matched_fields=["text"]
                ))
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                search_metadata={"search_mode": request.search_mode}
            )
        except Exception as e:
            logger.error(f"Text search error: {e}")
            raise HTTPException(status_code=500, detail="Text search failed")
    
    @app.post("/contexts/search/semantic", response_model=SearchResponse)
    async def search_semantic(
        request: SemanticSearchRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Search contexts using semantic similarity."""
        try:
            start_time = time.time()
            contexts = await server.context_store.search_semantic(
                query=request.query,
                namespaces=request.namespaces,
                similarity_threshold=request.similarity_threshold,
                limit=request.limit
            )
            search_time_ms = (time.time() - start_time) * 1000
            
            results = []
            for context in contexts:
                results.append(SearchResult(
                    context=_context_to_dict(context),
                    semantic_similarity=request.similarity_threshold,  # Placeholder
                    search_type=SearchType.SEMANTIC,
                    matched_fields=["text", "data"]
                ))
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                search_metadata={"similarity_threshold": request.similarity_threshold}
            )
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            raise HTTPException(status_code=500, detail="Semantic search failed")
    
    @app.post("/contexts/search/hybrid", response_model=List[Dict[str, Any]])
    async def search_hybrid(
        request: HybridSearchRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Hybrid search combining text search and metadata filters."""
        try:
            contexts = await server.context_store.search_hybrid(
                text_query=request.text_query,
                filters=request.filters,
                limit=request.limit
            )
            return [_context_to_dict(context) for context in contexts]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/contexts/search/hybrid_semantic", response_model=SearchResponse)
    async def search_hybrid_semantic(
        request: HybridSemanticSearchRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Hybrid search combining semantic similarity and text ranking."""
        try:
            start_time = time.time()
            # For local SQLite implementation, fall back to text search
            contexts = await server.context_store.search_text(
                query=request.query,
                namespaces=request.filters.namespaces if request.filters else None,
                limit=request.limit
            )
            search_time_ms = (time.time() - start_time) * 1000
            
            results = []
            for context in contexts:
                results.append(SearchResult(
                    context=_context_to_dict(context),
                    semantic_similarity=request.similarity_threshold,
                    text_rank=0.5,  # Placeholder
                    search_type=SearchType.HYBRID,
                    matched_fields=["text", "data"]
                ))
            
            return SearchResponse(
                results=results,
                total_found=len(results),
                search_time_ms=search_time_ms,
                search_metadata={
                    "search_strategy": request.search_strategy,
                    "semantic_weight": request.semantic_weight,
                    "text_weight": request.text_weight
                }
            )
        except Exception as e:
            logger.error(f"Hybrid semantic search error: {e}")
            raise HTTPException(status_code=500, detail="Hybrid semantic search failed")
    
    @app.post("/contexts/compose")
    async def compose_contexts(
        request: CompositionRequest,
        server: EpisodicServer = Depends(get_server),
        _: bool = Depends(verify_api_key)
    ):
        """Compose multiple contexts into a single context."""
        try:
            context = await server.context_store.compose(
                composition_id=request.composition_id,
                components=request.components,
                merge_strategy=request.merge_strategy
            )
            return _context_to_dict(context)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Embedding generation endpoint - matches episodic-cloud
    @app.post("/embeddings/generate", response_model=EmbeddingResponse)
    async def generate_embeddings(
        request: EmbeddingRequest,
        _: bool = Depends(verify_api_key)
    ):
        """Generate embeddings for given texts."""
        try:
            start_time = time.time()
            # For local implementation, return dummy embeddings
            # In a real implementation, this would use an embedding model
            embeddings = []
            for text in request.texts:
                # Generate dummy embedding of appropriate size
                embedding = [0.1] * 384  # all-MiniLM-L6-v2 has 384 dimensions
                embeddings.append(embedding)
            
            generation_time = time.time() - start_time
            
            return EmbeddingResponse(
                embeddings=embeddings,
                model=request.model,
                dimensions=384,
                generation_time_ms=generation_time * 1000
            )
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            raise HTTPException(status_code=500, detail="Embedding generation failed")
    
    @app.get("/health/search")
    async def health_check_search():
        """Health check for search functionality."""
        try:
            return {
                "status": "healthy",
                "search_engine": "local_sqlite",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Search health check error: {e}")
            raise HTTPException(status_code=503, detail="Search health check failed")
    
    # Subscription management endpoints - matches episodic-cloud interface
    @app.post("/subscriptions")
    async def create_subscription(
        subscription: Subscription,
        server: EpisodicServer = Depends(get_server)
    ):
        """Create a new subscription"""
        try:
            subscription_id = subscription.subscription_id or str(uuid.uuid4())
            
            # Store subscription in WebSocket manager
            server.websocket_manager.subscriptions[subscription_id] = {
                "subscription_id": subscription_id,
                "client_id": subscription.client_id,
                "delivery_method": subscription.delivery_method,
                "webhook_config": subscription.webhook_config.dict() if subscription.webhook_config else None,
                "filters": subscription.filters.dict()
            }
            
            return {
                "status": "created",
                "subscription_id": subscription_id
            }
        except Exception as e:
            logger.error(f"Subscription creation error: {e}")
            raise HTTPException(status_code=500, detail="Subscription creation failed")
    
    @app.delete("/subscriptions/{subscription_id}")
    async def delete_subscription(
        subscription_id: str,
        server: EpisodicServer = Depends(get_server)
    ):
        """Delete a subscription"""
        try:
            server.websocket_manager.subscriptions.pop(subscription_id, None)
            return {"status": "deleted", "subscription_id": subscription_id}
        except Exception as e:
            logger.error(f"Subscription deletion error: {e}")
            raise HTTPException(status_code=500, detail="Subscription deletion failed")
    
    @app.get("/subscriptions")
    async def list_subscriptions(
        client_id: Optional[str] = None,
        server: EpisodicServer = Depends(get_server)
    ):
        """List subscriptions"""
        try:
            subscriptions = []
            for sub_id, sub_data in server.websocket_manager.subscriptions.items():
                if client_id is None or sub_data["client_id"] == client_id:
                    subscriptions.append({
                        "subscription_id": sub_id,
                        "client_id": sub_data["client_id"],
                        "delivery_method": sub_data["delivery_method"],
                        "webhook_url": sub_data.get("webhook_config", {}).get("url") if sub_data.get("webhook_config") else None,
                        "filters": sub_data["filters"],
                        "created_at": datetime.utcnow().isoformat(),
                        "last_delivery_at": None,
                        "delivery_failures": 0
                    })
            return subscriptions
        except Exception as e:
            logger.error(f"Subscription listing error: {e}")
            raise HTTPException(status_code=500, detail="Subscription listing failed")
    
    # WebSocket endpoint for real-time subscriptions
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(
        websocket: WebSocket,
        client_id: str,
        server: EpisodicServer = Depends(get_server)
    ):
        """WebSocket endpoint for real-time context updates."""
        await server.websocket_manager.connect(websocket, client_id)
        try:
            while True:
                # Receive subscription messages from client
                data = await websocket.receive_text()
                try:
                    message = json.loads(data)
                    if message.get("type") == "subscribe":
                        namespace = message.get("namespace", "*")
                        server.websocket_manager.subscribe_to_namespace(client_id, namespace)
                        await websocket.send_text(json.dumps({
                            "type": "subscription_confirmed",
                            "namespace": namespace
                        }))
                    elif message.get("type") == "unsubscribe":
                        namespace = message.get("namespace", "*")
                        server.websocket_manager.unsubscribe_from_namespace(client_id, namespace)
                        await websocket.send_text(json.dumps({
                            "type": "unsubscription_confirmed",
                            "namespace": namespace
                        }))
                except json.JSONDecodeError:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }))
        except WebSocketDisconnect:
            server.websocket_manager.disconnect(client_id)
    
    return app


# For running the server directly
if __name__ == "__main__":
    import uvicorn
    import asyncio
    
    async def run_server():
        app = create_app()
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    try:
        asyncio.run(run_server())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            # If we're in a running event loop (e.g., Jupyter), create a new task
            loop = asyncio.get_event_loop()
            loop.create_task(run_server())
        else:
            raise 