#!/usr/bin/env python3
"""
Basic usage examples for the new Episodic architecture.
Demonstrates SqliteContextStore, Episodic client, and server usage.
"""

import asyncio
import json
import time
from episodic import SqliteContextStore, Episodic, ContextFilter


async def example_sqlite_store():
    """Example using SqliteContextStore directly (local storage)."""
    print("=== SqliteContextStore Example ===")
    
    # Create a local SQLite context store
    store = SqliteContextStore(db_path="./example_contexts.db")
    
    # Store some contexts
    await store.store(
        context_id="user_alice",
        data={"name": "Alice", "role": "admin", "last_login": time.time()},
        text="User Alice is an admin who last logged in recently",
        tags=["user", "admin"],
        namespace="users"
    )
    
    await store.store(
        context_id="user_bob", 
        data={"name": "Bob", "role": "user", "department": "engineering"},
        text="User Bob is an engineer",
        tags=["user", "engineering"],
        namespace="users"
    )
    
    # Retrieve a context
    alice = await store.get("user_alice")
    print(f"Retrieved: {alice.data}")
    
    # Query contexts
    user_filter = ContextFilter(namespaces=["users"], tags=["user"])
    users = await store.query(user_filter)
    print(f"Found {len(users)} users")
    
    # Search by text
    search_results = await store.search_text("admin")
    print(f"Text search for 'admin': {len(search_results)} results")
    
    # Clean up
    await store.close()
    print("SqliteContextStore example completed\n")


async def example_episodic_client():
    """Example using Episodic client (connects to remote server)."""
    print("=== Episodic Client Example ===")
    
    # This would connect to a remote Episodic server using ContextStoreClient
    # For this example, we'll show the API but not actually connect
    try:
        client = Episodic("https://your-episodic-server.com", api_key="your-api-key")
        
        # Store context
        await client.store(
            context_id="session_123",
            data={"user_id": "alice", "session_start": time.time()},
            text="Alice's current session",
            namespace="sessions"
        )
        
        # Get context
        session = await client.get("session_123")
        print(f"Session data: {session.data}")
        
        # Subscribe to updates (decorator style)
        @client.on_context_update(namespaces=["sessions"])
        async def handle_session_update(update):
            print(f"Session updated: {update.context.id}")
        
        print("Episodic client example completed")
        
    except Exception as e:
        print(f"Episodic client example skipped (no server): {e}")
    
    print()


def example_cli_usage():
    """Show CLI usage examples."""
    print("=== CLI Usage Examples ===")
    
    examples = [
        # Store a context
        'episodic store user_charlie \'{"name": "Charlie", "role": "manager"}\' --text "Charlie is a manager" --tags user manager --namespace users',
        
        # Store from file
        'episodic store config_prod @config.json --namespace configs --type configuration',
        
        # Get a context
        'episodic get user_charlie --format json',
        
        # Query contexts
        'episodic query --namespaces users --tags admin --limit 5 --format table',
        
        # Search contexts
        'episodic search text "manager" --namespaces users --format json',
        'episodic search semantic "user management" --threshold 0.8 --limit 3',
        
        # Delete a context
        'episodic delete user_charlie',
        
        # Run server
        'episodic server --host 0.0.0.0 --port 8000 --db-path ./production.db',
        
        # Health check
        'episodic health',
        
        # Using environment variables for remote server
        'EPISODIC_ENDPOINT=https://prod-server.com EPISODIC_API_KEY=secret episodic store ...',
    ]
    
    for example in examples:
        print(f"  {example}")
    
    print()


async def example_server_setup():
    """Example of setting up the server programmatically."""
    print("=== Server Setup Example ===")
    
    try:
        from episodic.server import create_app
        import uvicorn
        
        # Create the FastAPI app
        app = create_app(db_path="./server_contexts.db", namespace="production")
        
        print("Server app created successfully")
        print("To run: uvicorn app:app --host 0.0.0.0 --port 8000")
        print("Or use: episodic server --port 8000")
        
    except ImportError as e:
        print(f"Server setup example skipped (missing dependencies): {e}")
    
    print()


async def main():
    """Run all examples."""
    print("Episodic Context Store - New Architecture Examples")
    print("=" * 50)
    
    await example_sqlite_store()
    await example_episodic_client()
    example_cli_usage()
    await example_server_setup()
    
    print("All examples completed!")


if __name__ == "__main__":
    asyncio.run(main()) 