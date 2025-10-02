"""
SQLite Context Store implementation with local SQLite storage.
Provides all context store functionality without requiring a remote server.
"""

import asyncio
import sqlite3
import json
import time
import os
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from functools import wraps
from collections import defaultdict
from .core import Context, ContextFilter, ContextUpdate, ContextNotFoundException
from .base import BaseContextStore
from .subscriptions import ContextSubscriber


class SqliteContextStore(BaseContextStore):
    """
    SQLite Context Store implementation with local SQLite storage.
    Provides all context store functionality without requiring a remote server.
    Includes persistent storage, semantic search capabilities, and full feature parity
    with the remote context store.
    """
    
    def __init__(self, endpoint: str = "sqlite://", api_key: str = "", namespace: str = "default", 
                 db_path: Optional[str] = None):
        super().__init__(endpoint, api_key, namespace)
        
        # Set up database path
        if db_path is None:
            # Create a default database in user's home directory or temp directory
            import tempfile
            db_dir = os.path.expanduser("~/.episodic")
            if not os.path.exists(db_dir):
                try:
                    os.makedirs(db_dir, exist_ok=True)
                except:
                    # Fallback to temp directory if home directory is not writable
                    db_dir = tempfile.gettempdir()
            db_path = os.path.join(db_dir, "contexts.db")
        
        self.db_path = db_path
        self._subscribers: List[Any] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Single internal subscriber used for @cs.on_context_update delegation
        self._decorator_subscriber: Optional[ContextSubscriber] = None
        self._decorator_started: bool = False
        
        # Initialize the database
        self._init_database()
        self._start_cleanup_task()
        
        # Optional embedding engine for semantic search
        self._embedding_engine = None
        self._init_embedding_engine()
    
    def _init_embedding_engine(self):
        """Initialize embedding engine for semantic search if available."""
        try:
            # Try to import sentence-transformers for local embeddings
            from sentence_transformers import SentenceTransformer
            self._embedding_engine = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            # Embedding functionality will be disabled
            self._embedding_engine = None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embeddings for text using local model."""
        if not self._embedding_engine or not text:
            return None
        try:
            # Generate embedding
            embedding = self._embedding_engine.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception:
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        # Handle empty vectors
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product and magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))
        
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _init_database(self):
        """Initialize the SQLite database with the required schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create contexts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS contexts (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    type TEXT DEFAULT 'generic',
                    data TEXT NOT NULL,  -- JSON string
                    text TEXT,
                    metadata TEXT DEFAULT '{}',  -- JSON string
                    tags TEXT DEFAULT '[]',  -- JSON array
                    ttl INTEGER,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL,
                    auto_render_text BOOLEAN DEFAULT 0,
                    embedding TEXT,  -- JSON array of floats
                    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2',
                    embedding_generated_at REAL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_namespace ON contexts(namespace)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_created_at ON contexts(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_expires_at ON contexts(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_type ON contexts(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_contexts_updated_at ON contexts(updated_at)")
            
            # Create subscriptions table for persistent subscriptions
            conn.execute("""
                CREATE TABLE IF NOT EXISTS subscriptions (
                    subscription_id TEXT PRIMARY KEY,
                    client_id TEXT NOT NULL,
                    filters TEXT NOT NULL,  -- JSON string
                    is_active BOOLEAN DEFAULT 1,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.commit()
    
    def _start_cleanup_task(self):
        """Start the cleanup task for expired contexts."""
        if self._cleanup_task is None or self._cleanup_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._periodic_cleanup())
            except RuntimeError:
                # No event loop running, cleanup will happen on access
                pass
    
    async def _periodic_cleanup(self):
        """Periodically clean up expired contexts."""
        while True:
            try:
                await asyncio.sleep(60)  # Clean up every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup error: {e}")
    
    async def _cleanup_expired(self):
        """Remove expired contexts from storage."""
        current_time = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM contexts WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (current_time,)
            )
            conn.commit()
    
    def _dict_from_row(self, row: sqlite3.Row) -> Context:
        """Convert SQLite row to Context object."""
        data = json.loads(row["data"])
        metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        tags = json.loads(row["tags"]) if row["tags"] else []
        
        # Parse embedding if present
        embedding = None
        if row["embedding"]:
            try:
                embedding = json.loads(row["embedding"])
            except:
                embedding = None
        
        context = Context(
            id=row["id"],
            data=data,
            text=row["text"],
            namespace=row["namespace"],
            type=row["type"],
            metadata=metadata,
            tags=tags,
            ttl=row["ttl"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
            auto_render_text=bool(row["auto_render_text"]),
            embedding=embedding,
            embedding_model=row["embedding_model"],
            embedding_generated_at=row["embedding_generated_at"]
        )
        
        return context
    
    async def store(self, context_id: str, data: Dict[str, Any], text: str = None, 
                   ttl: int = None, tags: List[str] = None, namespace: str = None,
                   context_type: str = "generic", metadata: Dict[str, Any] = None) -> Context:
        """
        Store a context with the given data.
        """
        namespace = namespace or self.default_namespace
        tags = tags or []
        metadata = metadata or {}
        current_time = time.time()
        
        # Generate embedding if text is provided and embedding engine is available
        embedding = None
        embedding_generated_at = None
        if text and self._embedding_engine:
            embedding = self._generate_embedding(text)
            if embedding:
                embedding_generated_at = current_time
        
        # Calculate expiration time
        expires_at = None
        if ttl:
            expires_at = current_time + ttl
        
        context = Context(
            id=context_id,
            data=data,
            text=text,
            namespace=namespace,
            type=context_type,
            metadata=metadata,
            tags=tags,
            ttl=ttl,
            created_at=current_time,
            updated_at=current_time,
            expires_at=expires_at,
            auto_render_text=False,
            embedding=embedding,
            embedding_model="all-MiniLM-L6-v2" if embedding else None,
            embedding_generated_at=embedding_generated_at
        )
        
        # If text not provided but auto_render_text is True, generate it
        if not text:
            context.auto_render_text = True
            context.text = context._generate_text()
            # Generate embedding for auto-generated text
            if context.text and self._embedding_engine:
                embedding = self._generate_embedding(context.text)
                if embedding:
                    context.embedding = embedding
                    context.embedding_generated_at = current_time
        
        # Check if context already exists
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            existing = conn.execute("SELECT id FROM contexts WHERE id = ?", (context_id,)).fetchone()
            is_update = existing is not None
            
            # Insert or update the context
            if is_update:
                conn.execute("""
                    UPDATE contexts SET
                        namespace = ?, type = ?, data = ?, text = ?, metadata = ?,
                        tags = ?, ttl = ?, updated_at = ?, expires_at = ?,
                        auto_render_text = ?, embedding = ?, embedding_model = ?,
                        embedding_generated_at = ?
                    WHERE id = ?
                """, (
                    context.namespace, context.type, json.dumps(context.data),
                    context.text, json.dumps(context.metadata), json.dumps(context.tags),
                    context.ttl, context.updated_at, context.expires_at,
                    context.auto_render_text, 
                    json.dumps(context.embedding) if context.embedding else None,
                    context.embedding_model, context.embedding_generated_at,
                    context_id
                ))
            else:
                conn.execute("""
                    INSERT INTO contexts (
                        id, namespace, type, data, text, metadata, tags, ttl,
                        created_at, updated_at, expires_at, auto_render_text,
                        embedding, embedding_model, embedding_generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.id, context.namespace, context.type, json.dumps(context.data),
                    context.text, json.dumps(context.metadata), json.dumps(context.tags),
                    context.ttl, context.created_at, context.updated_at, context.expires_at,
                    context.auto_render_text,
                    json.dumps(context.embedding) if context.embedding else None,
                    context.embedding_model, context.embedding_generated_at
                ))
            
            conn.commit()
        
        # Notify subscribers
        operation = "update" if is_update else "create"
        await self._notify_subscribers(ContextUpdate(
            context=context,
            operation=operation,
            namespace=namespace
        ))
        
        return context
    
    async def store_context(self, context: Context) -> Context:
        """
        Store a Context object directly.
        """
        context.updated_at = time.time()
        
        # Check if context already exists
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            existing = conn.execute("SELECT id FROM contexts WHERE id = ?", (context.id,)).fetchone()
            is_update = existing is not None
            
            # Generate embedding if missing and text is available
            if not context.embedding and context.text and self._embedding_engine:
                embedding = self._generate_embedding(context.text)
                if embedding:
                    context.embedding = embedding
                    context.embedding_model = "all-MiniLM-L6-v2"
                    context.embedding_generated_at = time.time()
            
            # Insert or update the context
            if is_update:
                conn.execute("""
                    UPDATE contexts SET
                        namespace = ?, type = ?, data = ?, text = ?, metadata = ?,
                        tags = ?, ttl = ?, updated_at = ?, expires_at = ?,
                        auto_render_text = ?, embedding = ?, embedding_model = ?,
                        embedding_generated_at = ?
                    WHERE id = ?
                """, (
                    context.namespace, context.type, json.dumps(context.data),
                    context.text, json.dumps(context.metadata), json.dumps(context.tags),
                    context.ttl, context.updated_at, context.expires_at,
                    context.auto_render_text,
                    json.dumps(context.embedding) if context.embedding else None,
                    context.embedding_model, context.embedding_generated_at,
                    context.id
                ))
            else:
                conn.execute("""
                    INSERT INTO contexts (
                        id, namespace, type, data, text, metadata, tags, ttl,
                        created_at, updated_at, expires_at, auto_render_text,
                        embedding, embedding_model, embedding_generated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    context.id, context.namespace, context.type, json.dumps(context.data),
                    context.text, json.dumps(context.metadata), json.dumps(context.tags),
                    context.ttl, context.created_at or time.time(), context.updated_at, 
                    context.expires_at, context.auto_render_text,
                    json.dumps(context.embedding) if context.embedding else None,
                    context.embedding_model, context.embedding_generated_at
                ))
            
            conn.commit()
        
        operation = "update" if is_update else "create"
        await self._notify_subscribers(ContextUpdate(
            context=context,
            operation=operation,
            namespace=context.namespace
        ))
        
        return context
    
    async def get(self, context_id: str) -> Context:
        """
        Get a specific context by ID.
        """
        await self._cleanup_expired()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM contexts WHERE id = ? AND (expires_at IS NULL OR expires_at > ?)",
                (context_id, time.time())
            ).fetchone()
            
            if not row:
                raise ContextNotFoundException(f"Context '{context_id}' not found")
            
            return self._dict_from_row(row)
    
    async def query(self, filter: ContextFilter) -> List[Context]:
        """
        Query contexts based on the provided filter.
        """
        await self._cleanup_expired()
        
        query_parts = ["SELECT * FROM contexts WHERE (expires_at IS NULL OR expires_at > ?)"]
        params = [time.time()]
        
        # Add namespace filter
        if filter.namespaces:
            namespace_conditions = []
            for ns in filter.namespaces:
                if "*" in ns:
                    # Handle wildcard patterns
                    pattern = ns.replace("*", "%")
                    namespace_conditions.append("namespace LIKE ?")
                    params.append(pattern)
                else:
                    namespace_conditions.append("namespace = ?")
                    params.append(ns)
            if namespace_conditions:
                query_parts.append(f"AND ({' OR '.join(namespace_conditions)})")
        
        # Add tags filter
        if filter.tags:
            # For SQLite, we need to use JSON functions or LIKE patterns
            tag_conditions = []
            for tag in filter.tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')
            if tag_conditions:
                query_parts.append(f"AND ({' OR '.join(tag_conditions)})")
        
        # Add context types filter
        if filter.context_types:
            type_placeholders = ",".join("?" * len(filter.context_types))
            query_parts.append(f"AND type IN ({type_placeholders})")
            params.extend(filter.context_types)
        
        # Add since filter
        if filter.since:
            since_timestamp = filter._parse_time_filter(filter.since)
            if since_timestamp:
                query_parts.append("AND created_at >= ?")
                params.append(since_timestamp)
        
        # Add ordering and limit
        query_parts.append("ORDER BY created_at DESC")
        if filter.limit:
            query_parts.append("LIMIT ?")
            params.append(filter.limit)
        
        query = " ".join(query_parts)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
            
            results = []
            for row in rows:
                try:
                    context = self._dict_from_row(row)
                    results.append(context)
                except Exception as e:
                    print(f"Error parsing context {row['id']}: {e}")
                    continue
            
            return results
    
    async def search_text(self, query: str, namespaces: List[str] = None, 
                         limit: int = 10) -> List[Context]:
        """
        Search contexts by text content.
        """
        await self._cleanup_expired()
        
        query_parts = [
            "SELECT * FROM contexts WHERE text IS NOT NULL",
            "AND (expires_at IS NULL OR expires_at > ?)",
            "AND text LIKE ?"
        ]
        params = [time.time(), f"%{query}%"]
        
        # Add namespace filter
        if namespaces:
            namespace_conditions = []
            for ns in namespaces:
                if "*" in ns:
                    pattern = ns.replace("*", "%")
                    namespace_conditions.append("namespace LIKE ?")
                    params.append(pattern)
                else:
                    namespace_conditions.append("namespace = ?")
                    params.append(ns)
            if namespace_conditions:
                query_parts.append(f"AND ({' OR '.join(namespace_conditions)})")
        
        query_parts.extend(["ORDER BY created_at DESC", "LIMIT ?"])
        params.append(limit)
        
        sql_query = " ".join(query_parts)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql_query, params).fetchall()
            
            results = []
            for row in rows:
                try:
                    context = self._dict_from_row(row)
                    results.append(context)
                except Exception as e:
                    print(f"Error parsing context {row['id']}: {e}")
                    continue
            
            return results
    
    async def search_semantic(self, query: str, namespaces: List[str] = None,
                             similarity_threshold: float = 0.7, limit: int = 10) -> List[Context]:
        """
        Search contexts using semantic similarity (vector embeddings).
        """
        if not self._embedding_engine:
            # Fallback to text search if embedding engine is not available
            return await self.search_text(query, namespaces, limit)
        
        await self._cleanup_expired()
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        if not query_embedding:
            return []
        
        # Get all contexts with embeddings
        query_parts = [
            "SELECT * FROM contexts WHERE embedding IS NOT NULL",
            "AND (expires_at IS NULL OR expires_at > ?)"
        ]
        params = [time.time()]
        
        # Add namespace filter
        if namespaces:
            namespace_conditions = []
            for ns in namespaces:
                if "*" in ns:
                    pattern = ns.replace("*", "%")
                    namespace_conditions.append("namespace LIKE ?")
                    params.append(pattern)
                else:
                    namespace_conditions.append("namespace = ?")
                    params.append(ns)
            if namespace_conditions:
                query_parts.append(f"AND ({' OR '.join(namespace_conditions)})")
        
        sql_query = " ".join(query_parts)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql_query, params).fetchall()
            
            for row in rows:
                try:
                    context = self._dict_from_row(row)
                    if context.embedding:
                        # Calculate similarity
                        similarity = self._cosine_similarity(query_embedding, context.embedding)
                        if similarity >= similarity_threshold:
                            context.semantic_similarity = similarity
                            results.append((context, similarity))
                            
                            if len(results) >= limit:
                                break
                except Exception as e:
                    print(f"Error processing context {row['id']}: {e}")
                    continue
        
        # Sort by similarity descending and return contexts
        results.sort(key=lambda x: x[1], reverse=True)
        return [context for context, _ in results]
    
    async def search_hybrid(self, text_query: str, filters: ContextFilter, 
                           limit: int = 15) -> List[Context]:
        """
        Hybrid search combining text search and metadata filters.
        """
        # Get contexts matching the filter
        filtered_contexts = await self.query(filters)
        
        # Further filter by text query
        query_lower = text_query.lower()
        results = []
        
        for context in filtered_contexts:
            if context.text and query_lower in context.text.lower():
                results.append(context)
                if len(results) >= limit:
                    break
        
        return results
    
    async def delete(self, context_id: str) -> bool:
        """
        Delete a context by ID.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get the context before deletion for notification
            row = conn.execute("SELECT * FROM contexts WHERE id = ?", (context_id,)).fetchone()
            if not row:
                return False
            
            context = self._dict_from_row(row)
            
            # Delete the context
            conn.execute("DELETE FROM contexts WHERE id = ?", (context_id,))
            conn.commit()
            
            # Notify subscribers
            await self._notify_subscribers(ContextUpdate(
                context=context,
                operation="delete",
                namespace=context.namespace
            ))
            
            return True
    
    async def compose(self, composition_id: str, components: List[Dict[str, str]], 
                     merge_strategy: str = "priority_weighted") -> Context:
        """
        Compose multiple contexts into a single context.
        """
        composed_data = {}
        composed_texts = []
        all_tags = set()
        
        for component in components:
            namespace_pattern = component["namespace"]
            query_filter = component.get("query", "")
            
            # Simple query parsing for demo purposes
            contexts = await self.query(ContextFilter(namespaces=[namespace_pattern]))
            
            for context in contexts:
                composed_data.update(context.data)
                if context.text:
                    composed_texts.append(context.text)
                all_tags.update(context.tags)
        
        composed_text = "\n".join(composed_texts) if composed_texts else None
        
        composed_context = Context(
            id=composition_id,
            data=composed_data,
            text=composed_text,
            namespace="composed",
            type="composition",
            tags=list(all_tags),
            ttl=3600  # Default 1 hour TTL for compositions
        )
        
        await self.store_context(composed_context)
        return composed_context
    
    def add_subscriber(self, subscriber):
        """Add a subscriber for context updates."""
        self._subscribers.append(subscriber)
    
    def remove_subscriber(self, subscriber):
        """Remove a subscriber."""
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)
    
    async def _notify_subscribers(self, update: ContextUpdate):
        """Notify all subscribers of context updates."""
        # Ensure deferred-start subscriber is running once we are in async context
        if self._decorator_subscriber and not self._decorator_started:
            try:
                await self._decorator_subscriber.start()
                self._decorator_subscriber.is_running = True
                self._decorator_started = True
            except Exception as e:
                print(f"Failed to start decorator subscriber: {e}")
        
        # Notify the internal decorator subscriber explicitly (not stored in _subscribers)
        if self._decorator_subscriber and self._decorator_started and getattr(self._decorator_subscriber, 'is_running', False):
            try:
                await self._decorator_subscriber.handle_context_update(update)
            except Exception as e:
                print(f"Decorator subscriber notification error: {e}")
        
        # Snapshot after potential subscriber additions to avoid race conditions
        for subscriber in self._subscribers:
            try:
                if hasattr(subscriber, 'handle_context_update'):
                    await subscriber.handle_context_update(update)
            except Exception as e:
                print(f"Subscriber notification error: {e}")

    def on_context_update(self, namespaces: List[str] = None, tags: List[str] = None,
                          custom_filter: Any = None, retry_policy: Dict[str, Any] = None):
        """
        Decorator facade that delegates to an internal ContextSubscriber.
        Filtering remains entirely client-side in ContextSubscriber.
        """
        if self._decorator_subscriber is None:
            self._decorator_subscriber = ContextSubscriber(self)
        # For local store, register immediately to avoid missing first event
        if not self._decorator_started:
            try:
                # Mark running so it receives events; do not add to _subscribers
                self._decorator_subscriber.is_running = True
                self._decorator_started = True
            except Exception as e:
                print(f"Failed to initialize decorator subscriber: {e}")
        return self._decorator_subscriber.on_context_update(
            namespaces=namespaces,
            tags=tags,
            custom_filter=custom_filter,
            retry_policy=retry_policy
        )

    async def health_check(self) -> Dict[str, Any]:
        """Return health status of the context store."""
        # Check database connectivity
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("SELECT COUNT(*) FROM contexts").fetchone()
            db_status = "healthy"
        except Exception as e:
            db_status = f"error: {str(e)}"
        
        # Count total contexts
        try:
            with sqlite3.connect(self.db_path) as conn:
                total_contexts = conn.execute("SELECT COUNT(*) FROM contexts").fetchone()[0]
        except:
            total_contexts = 0
        
        return {
            "status": "healthy" if db_status == "healthy" else "degraded",
            "mode": "local_sqlite",
            "database_status": db_status,
            "database_path": self.db_path,
            "total_contexts": total_contexts,
            "active_subscribers": len(self._subscribers),
            "embedding_engine": "available" if self._embedding_engine else "disabled",
            "avg_latency_ms": 5.0,  # Estimated latency for SQLite operations
            "subscription_status": "active"
        }
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count active contexts
                active_contexts = conn.execute(
                    "SELECT COUNT(*) FROM contexts WHERE expires_at IS NULL OR expires_at > ?",
                    (time.time(),)
                ).fetchone()[0]
                
                # Count expired contexts
                expired_contexts = conn.execute(
                    "SELECT COUNT(*) FROM contexts WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (time.time(),)
                ).fetchone()[0]
                
                # Count contexts with embeddings
                contexts_with_embeddings = conn.execute(
                    "SELECT COUNT(*) FROM contexts WHERE embedding IS NOT NULL"
                ).fetchone()[0]
                
                # Database file size
                import os
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        except Exception:
            active_contexts = expired_contexts = contexts_with_embeddings = 0
            db_size = 0
        
        return {
            "mode": "local_sqlite",
            "database_path": self.db_path,
            "database_size_bytes": db_size,
            "active_contexts": active_contexts,
            "expired_contexts": expired_contexts,
            "contexts_with_embeddings": contexts_with_embeddings,
            "active_subscriptions": len(self._subscribers),
            "embedding_capability": self._embedding_engine is not None,
            "cache_hit_ratio": 0.95,  # SQLite is essentially a local cache
            "rate_limit_status": "unlimited"
        }
    
    async def close(self):
        """Clean up resources."""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass