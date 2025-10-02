"""
Subscription system for Context Store with decorator support.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Callable, Any, Set
from functools import wraps
from .core import ContextFilter, ContextUpdate, Context, SubscriptionException


class ContextSubscriber:
    """
    Context subscriber with decorator-based event handling.
    """
    
    def __init__(self, context_store, delivery_method: str = "direct"):
        self.context_store = context_store
        self.delivery_method = delivery_method
        self.handlers: List[Dict[str, Any]] = []
        self.batch_handlers: List[Dict[str, Any]] = []
        self.is_running = False
        self.batch_buffer: Dict[str, List[Context]] = {}
        self.batch_tasks: Dict[str, asyncio.Task] = {}
        
    def on_context_update(self, namespaces: List[str] = None, tags: List[str] = None,
                         custom_filter: Callable[[Context], bool] = None,
                         retry_policy: Dict[str, Any] = None):
        """
        Decorator for registering context update handlers.
        """
        def decorator(func):
            handler_info = {
                "function": func,
                "namespaces": namespaces or [],
                "tags": tags or [],
                "custom_filter": custom_filter,
                "retry_policy": retry_policy or {"max_retries": 0}
            }
            self.handlers.append(handler_info)
            return func
        return decorator
    
    def on_context_batch(self, namespaces: List[str] = None, batch_size: int = 50,
                        max_wait_ms: int = 5000):
        """
        Decorator for registering batch context handlers.
        """
        def decorator(func):
            handler_info = {
                "function": func,
                "namespaces": namespaces or [],
                "batch_size": batch_size,
                "max_wait_ms": max_wait_ms,
                "batch_key": f"{func.__name__}_{id(func)}"
            }
            self.batch_handlers.append(handler_info)
            return func
        return decorator
    
    async def start(self):
        """Start the subscriber and register with context store."""
        self.is_running = True
        
        # Try async version first, then fall back to sync
        if hasattr(self.context_store, 'add_subscriber_async'):
            await self.context_store.add_subscriber_async(self)
        elif asyncio.iscoroutinefunction(getattr(self.context_store, 'add_subscriber', None)):
            await self.context_store.add_subscriber(self)
        else:
            self.context_store.add_subscriber(self)
            
        print("Context subscriber started")
    
    async def stop(self):
        """Stop the subscriber."""
        self.is_running = False
        
        # Try async version first, then fall back to sync
        if hasattr(self.context_store, 'remove_subscriber_async'):
            await self.context_store.remove_subscriber_async(self)
        elif asyncio.iscoroutinefunction(getattr(self.context_store, 'remove_subscriber', None)):
            await self.context_store.remove_subscriber(self)
        else:
            self.context_store.remove_subscriber(self)
        
        # Cancel any pending batch tasks
        for task in self.batch_tasks.values():
            if not task.done():
                task.cancel()
        
        print("Context subscriber stopped")
    
    async def handle_context_update(self, update: ContextUpdate):
        """Handle context updates from the context store."""
        if not self.is_running:
            return
        
        context = update.context
        
        # Process regular handlers
        for handler_info in self.handlers:
            if self._matches_handler(context, handler_info):
                await self._execute_handler(handler_info, update)
        
        # Process batch handlers
        for handler_info in self.batch_handlers:
            if self._matches_batch_handler(context, handler_info):
                await self._add_to_batch(handler_info, context)
    
    def _matches_handler(self, context: Context, handler_info: Dict[str, Any]) -> bool:
        """Check if context matches handler criteria."""
        # Check namespaces
        if handler_info["namespaces"]:
            if not any(context.matches_namespace(ns) for ns in handler_info["namespaces"]):
                return False
        
        # Check tags
        if handler_info["tags"]:
            if not context.matches_tags(handler_info["tags"]):
                return False
        
        # Check custom filter
        if handler_info["custom_filter"]:
            try:
                if not handler_info["custom_filter"](context):
                    return False
            except Exception as e:
                print(f"Custom filter error: {e}")
                return False
        
        return True
    
    def _matches_batch_handler(self, context: Context, handler_info: Dict[str, Any]) -> bool:
        """Check if context matches batch handler criteria."""
        if handler_info["namespaces"]:
            return any(context.matches_namespace(ns) for ns in handler_info["namespaces"])
        return True
    
    async def _execute_handler(self, handler_info: Dict[str, Any], update: ContextUpdate):
        """Execute a context handler with retry logic."""
        func = handler_info["function"]
        retry_policy = handler_info["retry_policy"]
        max_retries = retry_policy.get("max_retries", 0)
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(update)
                else:
                    func(update)
                break
            except Exception as e:
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Handler {func.__name__} failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Handler {func.__name__} failed after {max_retries + 1} attempts: {e}")
    
    async def _add_to_batch(self, handler_info: Dict[str, Any], context: Context):
        """Add context to batch buffer and schedule processing."""
        batch_key = handler_info["batch_key"]
        
        if batch_key not in self.batch_buffer:
            self.batch_buffer[batch_key] = []
        
        self.batch_buffer[batch_key].append(context)
        
        # Check if batch is full
        if len(self.batch_buffer[batch_key]) >= handler_info["batch_size"]:
            await self._process_batch(handler_info, batch_key)
        else:
            # Schedule batch processing with timeout
            if batch_key not in self.batch_tasks or self.batch_tasks[batch_key].done():
                self.batch_tasks[batch_key] = asyncio.create_task(
                    self._batch_timeout(handler_info, batch_key, handler_info["max_wait_ms"] / 1000)
                )
    
    async def _batch_timeout(self, handler_info: Dict[str, Any], batch_key: str, timeout: float):
        """Process batch after timeout."""
        await asyncio.sleep(timeout)
        await self._process_batch(handler_info, batch_key)
    
    async def _process_batch(self, handler_info: Dict[str, Any], batch_key: str):
        """Process a batch of contexts."""
        if batch_key not in self.batch_buffer or not self.batch_buffer[batch_key]:
            return
        
        contexts = self.batch_buffer[batch_key]
        self.batch_buffer[batch_key] = []
        
        # Cancel timeout task if running
        if batch_key in self.batch_tasks and not self.batch_tasks[batch_key].done():
            self.batch_tasks[batch_key].cancel()
        
        try:
            func = handler_info["function"]
            if asyncio.iscoroutinefunction(func):
                await func(contexts)
            else:
                func(contexts)
        except Exception as e:
            print(f"Batch handler {func.__name__} failed: {e}")


class ActionHandler:
    """
    Enhanced handler for executing actions based on context updates.
    """
    
    def __init__(self, context_store):
        self.context_store = context_store
        self.action_handlers: Dict[str, Callable] = {}
    
    def register_action(self, namespace_pattern: str, handler: Callable):
        """Register an action handler for a namespace pattern."""
        self.action_handlers[namespace_pattern] = handler
    
    async def execute_action(self, context: Context) -> Any:
        """Execute the appropriate action for a context."""
        for pattern, handler in self.action_handlers.items():
            if context.matches_namespace(pattern):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        result = await handler(context, self.context_store)
                    else:
                        result = handler(context, self.context_store)
                    
                    # Store action result
                    await self._store_action_result(context, handler.__name__, result)
                    return result
                except Exception as e:
                    await self._store_action_error(context, handler.__name__, str(e))
                    raise
        
        return None
    
    async def _store_action_result(self, original_context: Context, 
                                  action_name: str, result: Any):
        """Store the result of an action execution."""
        action_context = Context(
            id=f"actions.{action_name}.{original_context.id}.{int(time.time())}",
            data={
                "original_context_id": original_context.id,
                "action_name": action_name,
                "result": result,
                "timestamp": time.time()
            },
            text=f"Action {action_name} executed for {original_context.id}",
            namespace="actions",
            type="action_result",
            tags=["action", "result", action_name.lower()],
            ttl=3600
        )
        
        await self.context_store.store_context(action_context)
    
    async def _store_action_error(self, original_context: Context,
                                 action_name: str, error: str):
        """Store action execution errors."""
        error_context = Context(
            id=f"actions.errors.{action_name}.{original_context.id}.{int(time.time())}",
            data={
                "original_context_id": original_context.id,
                "action_name": action_name,
                "error": error,
                "timestamp": time.time()
            },
            text=f"Action {action_name} failed for {original_context.id}: {error}",
            namespace="actions.errors",
            type="action_error",
            tags=["action", "error", action_name.lower()],
            ttl=86400
        )
        
        await self.context_store.store_context(error_context)


class WebSocketSubscription:
    """
    Direct WebSocket-based subscription handler for standalone connections.
    This creates a direct WebSocket connection to the server, independent of the context store.
    """
    
    def __init__(self, websocket_url: str, api_key: str = None):
        self.websocket_url = websocket_url
        self.api_key = api_key
        self.filters: List[ContextFilter] = []
        self.action_handlers: Dict[str, Callable] = {}
        self.websocket = None
        self.is_running = False
        self._websocket_task: Optional[asyncio.Task] = None
        self._reconnect_delay: float = 1.0
        self._max_reconnect_delay: float = 30.0
    
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            import websockets
            
            additional_headers = {}
            if self.api_key:
                additional_headers["X-API-Key"] = self.api_key
            
            # Connect with proper headers parameter for websockets 15.x
            if additional_headers:
                self.websocket = await websockets.connect(
                    self.websocket_url,
                    additional_headers=additional_headers,
                    ping_interval=30,
                    ping_timeout=10
                )
            else:
                self.websocket = await websockets.connect(
                    self.websocket_url,
                    ping_interval=30,
                    ping_timeout=10
                )
            
            self.is_running = True
            self._reconnect_delay = 1.0  # Reset reconnect delay
            
            # Start listening for messages
            if self._websocket_task is None or self._websocket_task.done():
                self._websocket_task = asyncio.create_task(self._websocket_listener())
            
            print(f"WebSocket connected to {self.websocket_url}")
            return True
            
        except Exception as e:
            print(f"Failed to connect WebSocket: {e}")
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self.is_running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                if self._websocket_task and not self._websocket_task.done():
                    self._websocket_task.cancel()
            except Exception as e:
                print(f"Error closing WebSocket: {e}")
            finally:
                self.websocket = None
    
    async def subscribe(self, context_filter: ContextFilter, action_handler: Callable = None):
        """Subscribe to context updates with optional action handler."""
        self.filters.append(context_filter)
        
        if action_handler:
            handler_key = f"filter_{len(self.filters) - 1}"
            self.action_handlers[handler_key] = action_handler
        
        # Send subscription message if connected
        if self.websocket and self.is_running:
            try:
                subscribe_message = {
                    "type": "subscribe",
                    "filters": {
                        "namespaces": context_filter.namespaces,
                        "tags": context_filter.tags,
                        "context_types": context_filter.context_types,
                        "since": context_filter.since,
                        "limit": context_filter.limit,
                        "include_expired": context_filter.include_expired
                    }
                }
                await self.websocket.send(json.dumps(subscribe_message))
                print(f"Subscription sent for filter: {context_filter.namespaces}")
            except Exception as e:
                print(f"Failed to send subscription: {e}")
    
    async def _websocket_listener(self):
        """Listen for WebSocket messages."""
        try:
            while self.websocket and self.is_running:
                try:
                    message = await self.websocket.recv()
                    await self._handle_websocket_message(message)
                except Exception as e:
                    if self.is_running:  # Only log if we're supposed to be running
                        print(f"WebSocket receive error: {e}")
                    break
        except Exception as e:
            if self.is_running:  # Only log if we're supposed to be running
                print(f"WebSocket listener error: {e}")
        finally:
            if self.is_running:  # Only reconnect if we're supposed to be running
                await self._schedule_reconnect()
    
    async def _handle_websocket_message(self, message: str):
        """Handle incoming WebSocket messages."""
        try:
            data = json.loads(message)
            
            if data.get("type") == "context_update":
                # Convert to Context and ContextUpdate objects
                context_data = data.get("context", {})
                context = self._context_from_dict(context_data)
                
                update = ContextUpdate(
                    context=context,
                    operation=data.get("operation", "update"),
                    namespace=context.namespace
                )
                
                # Check filters and execute matching handlers
                for i, context_filter in enumerate(self.filters):
                    if context_filter.matches(context):
                        handler_key = f"filter_{i}"
                        if handler_key in self.action_handlers:
                            try:
                                handler = self.action_handlers[handler_key]
                                if asyncio.iscoroutinefunction(handler):
                                    await handler(update)
                                else:
                                    handler(update)
                            except Exception as e:
                                print(f"Action handler error: {e}")
                        break
                        
            elif data.get("type") == "subscription_confirmed":
                print("Subscription confirmed")
                
            elif data.get("type") == "error":
                print(f"WebSocket error: {data.get('message', 'Unknown error')}")
                
        except Exception as e:
            print(f"Error handling WebSocket message: {e}")
    
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
            auto_render_text=data.get("auto_render_text", False)
        )
    
    async def _schedule_reconnect(self):
        """Schedule WebSocket reconnection with exponential backoff."""
        if not self.is_running:
            return
        
        await asyncio.sleep(self._reconnect_delay)
        
        if await self.connect():
            # Re-subscribe all filters
            for context_filter in self.filters:
                await self.subscribe(context_filter)
        else:
            # Increase reconnect delay with exponential backoff
            self._reconnect_delay = min(self._reconnect_delay * 2, self._max_reconnect_delay)
            await self._schedule_reconnect()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()