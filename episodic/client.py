"""
Episodic client for connecting to remote Context Store endpoints.
Includes the base ContextStoreClient and the main Episodic client class.
"""

import json
import time
import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Any, Callable
from urllib.parse import urljoin, urlparse
import logging

from .base import BaseContextStore
from .core import Context, ContextFilter, ContextUpdate, ContextNotFoundException, ContextStoreException
from .subscriptions import ContextSubscriber

logger = logging.getLogger(__name__)


def _is_valid_http_endpoint(endpoint: str) -> bool:
    """Check if the endpoint is a valid HTTP/HTTPS URL."""
    if not endpoint:
        return False
    
    try:
        parsed = urlparse(endpoint)
        return parsed.scheme in ('http', 'https') and bool(parsed.netloc)
    except Exception:
        return False


class ContextStoreClient(BaseContextStore):
    """
    Context Store client implementation that communicates with a server via HTTP REST API.
    Supports real-time WebSocket subscriptions for context updates.
    """
    
    def __init__(self, endpoint: str, api_key: str = "", namespace: str = "default", timeout: int = 30):
        super().__init__(endpoint, api_key, namespace)
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        
        # WebSocket support
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._client_id: str = f"client_{int(time.time() * 1000)}"
        self._websocket_url: str = self._build_websocket_url(endpoint)
        self._subscribers: List[Any] = []
        self._websocket_task: Optional[asyncio.Task] = None
        self._is_connected: bool = False
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 30.0
        # Single internal subscriber used for @cs.on_context_update delegation
        self._decorator_subscriber: Optional[ContextSubscriber] = None
        self._decorator_started: bool = False
        
    
    def _build_websocket_url(self, endpoint: str) -> str:
        """Convert HTTP endpoint to WebSocket URL."""
        parsed = urlparse(endpoint.rstrip('/'))
        ws_scheme = 'wss' if parsed.scheme == 'https' else 'ws'
        return f"{ws_scheme}://{parsed.netloc}/ws/{self._client_id}"
    
    async def _connect_websocket(self) -> bool:
        """Establish WebSocket connection."""
        try:
            # Prepare headers for authentication
            additional_headers = {}
            if self.api_key:
                additional_headers["X-API-Key"] = self.api_key
            
            # Connect with proper headers parameter for websockets 15.x
            if additional_headers:
                self._websocket = await websockets.connect(
                    self._websocket_url,
                    additional_headers=additional_headers,
                    ping_interval=30,
                    ping_timeout=10
                )
            else:
                self._websocket = await websockets.connect(
                    self._websocket_url,
                    ping_interval=30,
                    ping_timeout=10
                )
            self._is_connected = True
            self._reconnect_delay = 1.0  # Reset reconnect delay on successful connection
            
            logger.info(f"WebSocket connected to {self._websocket_url}")
            
            # Start listening for messages
            if self._websocket_task is None or self._websocket_task.done():
                self._websocket_task = asyncio.create_task(self._websocket_listener())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            self._is_connected = False
            return False
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages and handle them."""
        try:
            while self._websocket and self._is_connected:
                try:
                    message = await self._websocket.recv()
                    await self._handle_websocket_message(message)
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    break
                except websockets.exceptions.ConnectionClosedError:
                    logger.warning("WebSocket connection closed with error")
                    break
                except Exception as e:
                    logger.error(f"Error receiving WebSocket message: {e}")
                    break
        except Exception as e:
            logger.error(f"WebSocket listener error: {e}")
        finally:
            self._is_connected = False
            await self._schedule_reconnect()
    
    async def _handle_websocket_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            if data.get("type") == "context_update":
                # Convert to ContextUpdate and notify subscribers
                # The server sends: { type: "context_update", data: { context: {...}, operation: "...", namespace: "...", timestamp: ... } }
                message_data = data.get("data", {})
                context_data = message_data.get("context", {})
                context = self._context_from_dict(context_data)
                
                update = ContextUpdate(
                    context=context,
                    operation=message_data.get("operation", "update"),
                    namespace=context.namespace
                )
                
                await self._notify_subscribers(update)
                
            elif data.get("type") == "subscription_confirmed":
                logger.info(f"Subscription confirmed for client {self._client_id}")
                
            elif data.get("type") == "error":
                logger.error(f"WebSocket error: {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def _schedule_reconnect(self):
        """Schedule WebSocket reconnection with exponential backoff."""
        if not self._subscribers:
            return  # No need to reconnect if no subscribers
        
        await asyncio.sleep(self._reconnect_delay)
        
        if await self._connect_websocket():
            # Re-subscribe if we have subscribers
            await self._resubscribe_on_reconnect()
        else:
            # Increase reconnect delay with exponential backoff
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            await self._schedule_reconnect()
    
    async def _resubscribe_on_reconnect(self):
        """Re-establish subscriptions after reconnection."""
        if self._websocket and self._is_connected and self._subscribers:
            try:
                subscribe_message = {
                    "type": "subscribe",
                    "filters": {
                        # Subscribe to all namespaces by default; client-side filters will apply
                        "limit": 100
                    }
                }
                await self._websocket.send(json.dumps(subscribe_message))
                logger.info("Re-subscribed after reconnection")
            except Exception as e:
                logger.error(f"Failed to re-subscribe after reconnection: {e}")
    
    async def _notify_subscribers(self, update: ContextUpdate):
        """Notify all subscribers of context updates."""
        # Ensure internal decorator subscriber is started when notifications begin
        if self._decorator_subscriber and not self._decorator_started:
            try:
                await self._decorator_subscriber.start()
                self._decorator_started = True
            except Exception as e:
                logger.error(f"Failed to start decorator subscriber: {e}")
        for subscriber in self._subscribers:
            try:
                if hasattr(subscriber, 'handle_context_update'):
                    await subscriber.handle_context_update(update)
            except Exception as e:
                logger.error(f"Subscriber notification error: {e}")
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            headers = {}
            if self.api_key:
                headers["X-API-Key"] = self.api_key
            headers["Content-Type"] = "application/json"
            
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
        return self._session
    
    async def _make_request(self, method: str, path: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to the server."""
        session = await self._get_session()
        url = urljoin(self.endpoint.rstrip('/') + '/', path.lstrip('/'))
        
        try:
            if method.upper() == "GET":
                async with session.get(url) as response:
                    return await self._handle_response(response)
            elif method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    return await self._handle_response(response)
            elif method.upper() == "DELETE":
                async with session.delete(url) as response:
                    return await self._handle_response(response)
            else:
                raise ContextStoreException(f"Unsupported HTTP method: {method}")
        except aiohttp.ClientError as e:
            raise ContextStoreException(f"HTTP request failed: {str(e)}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Dict:
        """Handle HTTP response and raise appropriate exceptions."""
        try:
            response_data = await response.json()
        except json.JSONDecodeError:
            response_data = {"detail": await response.text()}
        
        if response.status == 404:
            raise ContextNotFoundException(response_data.get("detail", "Context not found"))
        elif response.status >= 400:
            raise ContextStoreException(f"HTTP {response.status}: {response_data.get('detail', 'Unknown error')}")
        
        return response_data
    
    def _context_from_dict(self, data: Dict) -> Context:
        """Convert dictionary response to Context object."""
        return Context(
            id=data["id"],
            data=data["data"],
            text=data.get("text"),
            namespace=data["namespace"],
            type=data["type"],
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            ttl=data.get("ttl"),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            expires_at=data.get("expires_at"),
            auto_render_text=data.get("auto_render_text", False),
            # embedding=data.get("embedding"),
            embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
            embedding_generated_at=data.get("embedding_generated_at"),
            semantic_similarity=data.get("semantic_similarity"),
            text_rank=data.get("text_rank"),
            hybrid_score=data.get("hybrid_score")
        )
    
    async def store(self, context_id: str, data: Dict[str, Any], text: str = None, 
                   ttl: int = None, tags: List[str] = None, namespace: str = None,
                   context_type: str = "generic", metadata: Dict[str, Any] = None) -> Context:
        """Store a context with the given data."""
        request_data = {
            "id": context_id,
            "data": data,
            "text": text,
            "ttl": ttl,
            "tags": tags or [],
            "namespace": namespace or self.default_namespace,
            "type": context_type,
            "metadata": metadata or {}
        }
        
        response_data = await self._make_request("POST", "/contexts", request_data)
        return self._context_from_dict(response_data)
    
    async def store_context(self, context: Context) -> Context:
        """Store a Context object directly."""
        request_data = {
            "id": context.id,
            "data": context.data,
            "text": context.text,
            "namespace": context.namespace,
            "type": context.type,
            "metadata": context.metadata,
            "tags": context.tags,
            "ttl": context.ttl,
            "created_at": context.created_at,
            "updated_at": context.updated_at,
            "expires_at": context.expires_at,
            "auto_render_text": context.auto_render_text
        }
        
        response_data = await self._make_request("POST", "/contexts/object", request_data)
        return self._context_from_dict(response_data)
    
    async def get(self, context_id: str) -> Context:
        """Get a specific context by ID."""
        response_data = await self._make_request("GET", f"/contexts/{context_id}")
        return self._context_from_dict(response_data)
    
    async def query(self, filter: ContextFilter) -> List[Context]:
        """Query contexts based on the provided filter."""
        request_data = {
            "namespaces": filter.namespaces,
            "tags": filter.tags,
            "context_types": filter.context_types,
            "since": filter.since,
            "limit": filter.limit,
            "include_expired": filter.include_expired
        }
        
        response_data = await self._make_request("POST", "/contexts/query", request_data)
        return [self._context_from_dict(ctx_data) for ctx_data in response_data]
    
    async def search_text(self, query: str, namespaces: List[str] = None, 
                         limit: int = 10) -> List[Context]:
        """Search contexts by text content."""
        request_data = {
            "query": query,
            "namespaces": namespaces,
            "limit": limit
        }

        response_data = await self._make_request("POST", "/contexts/search/text", request_data)

        # Handle SearchResponse format from the Supabase-backed service
        if isinstance(response_data, dict) and "results" in response_data:
            contexts: List[Context] = []
            for result in response_data["results"]:
                context_dict = result.get("context", result)
                context = self._context_from_dict(context_dict)

                # Capture optional ranking metadata when present
                if "text_rank" in result:
                    context.text_rank = result["text_rank"]
                if "hybrid_score" in result:
                    context.hybrid_score = result["hybrid_score"]

                contexts.append(context)
            return contexts

        return [self._context_from_dict(ctx_data) for ctx_data in response_data]
    
    async def search_semantic(self, query: str, namespaces: List[str] = None,
                             similarity_threshold: float = 0.7, limit: int = 10) -> List[Context]:
        """Search contexts using semantic similarity (vector embeddings)."""
        request_data = {
            "query": query,
            "namespaces": namespaces,
            "similarity_threshold": similarity_threshold,
            "limit": limit
        }
        
        response_data = await self._make_request("POST", "/contexts/search/semantic", request_data)
        
        # Handle SearchResponse format
        if isinstance(response_data, dict) and "results" in response_data:
            contexts = []
            for result in response_data["results"]:
                context_dict = result.get("context", result)
                context = self._context_from_dict(context_dict)
                # Add semantic similarity score if available
                if "semantic_similarity" in result:
                    context.semantic_similarity = result["semantic_similarity"]
                contexts.append(context)
            return contexts
        else:
            return [self._context_from_dict(ctx_data) for ctx_data in response_data]
    
    async def search_hybrid(self, text_query: str, filters: ContextFilter, 
                           limit: int = 15) -> List[Context]:
        """Hybrid search combining text search and metadata filters."""
        request_data = {
            "text_query": text_query,
            "filters": {
                "namespaces": filters.namespaces,
                "tags": filters.tags,
                "context_types": filters.context_types,
                "since": filters.since,
                "limit": filters.limit,
                "include_expired": filters.include_expired
            },
            "limit": limit
        }
        
        response_data = await self._make_request("POST", "/contexts/search/hybrid", request_data)
        return [self._context_from_dict(ctx_data) for ctx_data in response_data]
    
    async def delete(self, context_id: str) -> bool:
        """Delete a context by ID."""
        try:
            await self._make_request("DELETE", f"/contexts/{context_id}")
            return True
        except ContextNotFoundException:
            return False
    
    async def compose(self, composition_id: str, components: List[Dict[str, str]], 
                     merge_strategy: str = "priority_weighted") -> Context:
        """Compose multiple contexts into a single context."""
        request_data = {
            "composition_id": composition_id,
            "components": components,
            "merge_strategy": merge_strategy
        }
        
        response_data = await self._make_request("POST", "/contexts/compose", request_data)
        return self._context_from_dict(response_data)
    
    def add_subscriber(self, subscriber):
        """Add a subscriber for context updates (sync version for compatibility)."""
        self._subscribers.append(subscriber)
        
        # Connect WebSocket if this is the first subscriber
        if len(self._subscribers) == 1 and not self._is_connected:
            # Schedule WebSocket connection in the background
            asyncio.create_task(self._connect_and_subscribe())
    
    def remove_subscriber(self, subscriber):
        """Remove a subscriber (sync version for compatibility)."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            
        # Close WebSocket if no more subscribers
        if not self._subscribers and self._websocket:
            asyncio.create_task(self._disconnect_websocket())
    
    async def add_subscriber_async(self, subscriber):
        """Add a subscriber for context updates (async version)."""
        self._subscribers.append(subscriber)
        
        # Connect WebSocket if this is the first subscriber
        if len(self._subscribers) == 1 and not self._is_connected:
            await self._connect_and_subscribe()
    
    async def remove_subscriber_async(self, subscriber):
        """Remove a subscriber (async version)."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
            
        # Close WebSocket if no more subscribers
        if not self._subscribers and self._websocket:
            await self._disconnect_websocket()
    
    async def _connect_and_subscribe(self):
        """Helper to connect WebSocket and send subscription message."""
        await self._connect_websocket()
        
        # Send subscription message
        if self._websocket and self._is_connected:
            try:
                subscribe_message = {
                    "type": "subscribe",
                    "filters": {
                        # Subscribe to all namespaces by default; client-side filters will apply
                        "limit": 100
                    }
                }
                await self._websocket.send(json.dumps(subscribe_message))
            except Exception as e:
                logger.error(f"Failed to send subscription message: {e}")

    def on_context_update(self, namespaces: List[str] = None, tags: List[str] = None,
                          custom_filter: Any = None, retry_policy: Dict[str, Any] = None):
        """
        Decorator facade that delegates to an internal ContextSubscriber.
        Filtering remains entirely client-side in ContextSubscriber.
        """
        if self._decorator_subscriber is None:
            self._decorator_subscriber = ContextSubscriber(self)
        return self._decorator_subscriber.on_context_update(
            namespaces=namespaces,
            tags=tags,
            custom_filter=custom_filter,
            retry_policy=retry_policy
        )
    
    async def _disconnect_websocket(self):
        """Helper to disconnect WebSocket."""
        try:
            if self._websocket:
                await self._websocket.close()
            if self._websocket_task and not self._websocket_task.done():
                self._websocket_task.cancel()
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        finally:
            self._websocket = None
            self._is_connected = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the context store."""
        # Validate API key first; will raise ContextStoreException (e.g., 403) if invalid
        await self._make_request("GET", "/auth/validate")

        # Then fetch health
        response_data = await self._make_request("GET", "/health")
        response_data["mode"] = "remote"
        return response_data
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        response_data = await self._make_request("GET", "/diagnostics")
        response_data["mode"] = "remote"
        return response_data
    
    async def close(self):
        """Close the HTTP session and WebSocket connection."""
        # Close WebSocket
        if self._websocket:
            try:
                await self._websocket.close()
                if self._websocket_task and not self._websocket_task.done():
                    self._websocket_task.cancel()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
            finally:
                self._websocket = None
                self._is_connected = False
        
        # Close HTTP session
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


class Episodic(ContextStoreClient):
    """
    Episodic client for connecting to remote Context Store endpoints.
    
    This is the main client class for interacting with remote Episodic Context Store servers.
    It provides a clean interface for storing, retrieving, and subscribing to context updates
    over HTTP/HTTPS with WebSocket support for real-time subscriptions.
    
    Example:
        ```python
        # Connect to a remote Episodic server
        client = Episodic("https://your-episodic-server.com", api_key="your-api-key")
        
        # Store context
        await client.store("user_123", {"name": "Alice", "role": "admin"})
        
        # Retrieve context
        context = await client.get("user_123")
        
        # Subscribe to updates
        @client.on_context_update(namespaces=["users"])
        async def handle_update(update):
            print(f"Context updated: {update.context.id}")
        ```
    """
    
    def __init__(self, endpoint: str, api_key: str = "", namespace: str = "default", timeout: int = 30):
        """
        Initialize the Episodic client.
        
        Args:
            endpoint: The HTTP/HTTPS endpoint of the Episodic server
            api_key: Optional API key for authentication
            namespace: Default namespace for contexts
            timeout: Request timeout in seconds
        """
        if not _is_valid_http_endpoint(endpoint):
            raise ValueError(f"Invalid HTTP endpoint: {endpoint}. Episodic client requires HTTP/HTTPS URLs.")
        
        super().__init__(endpoint, api_key, namespace, timeout)


# Backward compatibility alias - ContextStore is exactly the same as Episodic
class ContextStore(Episodic):
    """
    Backward compatibility alias for Episodic client.
    
    ContextStore is exactly the same as Episodic - it's a client for connecting to remote
    Context Store endpoints via HTTP/HTTPS with WebSocket support.
    
    Note: This is provided for backward compatibility. New code should use Episodic directly.
    
    Example:
        ```python
        # Backward compatible usage
        client = ContextStore("https://your-episodic-server.com", api_key="your-api-key")
        
        # This is equivalent to:
        client = Episodic("https://your-episodic-server.com", api_key="your-api-key")
        ```
    """
    pass


__all__ = ["ContextStoreClient", "Episodic", "ContextStore"]
