"""
Episodic Python SDK - Context Store for AI Applications

A clean implementation of the Episodic context store with clear client-server separation:
- SqliteContextStore: Local SQLite-based storage backend
- Episodic: HTTP/WebSocket client for remote servers
- Server: FastAPI server implementation
"""

from .core import (
    Context,
    ContextFilter,
    ContextUpdate,
    ContextStoreException,
    ContextNotFoundException,
    SubscriptionException
)

from .base import BaseContextStore
from .store import SqliteContextStore
from .client import Episodic, ContextStoreClient, ContextStore

from .subscriptions import (
    ContextSubscriber,
    ActionHandler,
    WebSocketSubscription
)

__version__ = "0.1.0"
__all__ = [
    "Context",
    "ContextFilter", 
    "ContextUpdate",
    "BaseContextStore",
    "SqliteContextStore",
    "Episodic",
    "ContextStoreClient",
    "ContextStore",  # Backward compatibility alias
    "ContextSubscriber",
    "ActionHandler",
    "WebSocketSubscription",
    "ContextStoreException",
    "ContextNotFoundException",
    "SubscriptionException"
]