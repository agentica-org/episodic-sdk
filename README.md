<div align="center">

# Episodic

</div>

<h3 align="center">
Context Store for AI Agents
</h3>

Episodic SDK provides a flexible context storage system for building context-aware AI agents with a client-server architecture.

## Quick Start

### Installation

Install with semantic search capabilities (recommended):

```bash
pip install -e .[semantic]
```

Or install without semantic search:
```bash
pip install -e .
```

### Running the Server

Start the Context Store server:

```bash
episodic serve --port 8000
```

Or use uvicorn directly:
```bash
uvicorn episodic.server:app --host 0.0.0.0 --port 8000
```

The server will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs

### Using the Client

Connect to a remote Context Store server:

```python
import asyncio
from episodic import ContextStore, ContextFilter

async def main():
    # Initialize the HTTP client
    client = ContextStore(
        endpoint="http://localhost:8000",
        api_key="your-api-key",
        namespace="my-app"
    )
    
    try:
        # Store a context
        await client.store(
            context_id="weather.sf.current",
            data={"temperature": 72, "conditions": "sunny"},
            text="Current weather in San Francisco: 72°F, sunny",
            ttl=1800,  # 30 minutes
            tags=["weather", "real-time"]
        )
        
        # Retrieve the context
        context = await client.get("weather.sf.current")
        print(f"Temperature: {context.data['temperature']}°F")
        
        # Query contexts with filters
        results = await client.query(
            ContextFilter(tags=["weather"], limit=10)
        )
        
        # Search by text
        results = await client.search_text("sunny weather")
        
    finally:
        await client.close()

asyncio.run(main())
```

## Key Features

- **Client-Server Architecture**: Scalable distributed deployment with FastAPI
- **Semantic Search**: Optional vector embeddings for similarity search
- **WebSocket Subscriptions**: Real-time updates for context changes
- **Flexible Querying**: Filter by namespace, tags, time ranges, and more
- **Context Composition**: Merge multiple contexts with custom strategies
- **RESTful API**: Full-featured HTTP API with automatic documentation


## Core Operations

### Querying Contexts

```python
from episodic import ContextFilter

# Query with filters
contexts = await client.query(
    ContextFilter(
        tags=["temperature"],
        since="1h",  # Last hour
        limit=50
    )
)
```

### Real-time Subscriptions

```python
from episodic import ContextSubscriber

subscriber = ContextSubscriber(context_store)

@subscriber.on_context_update(tags=["important"])
async def handle_updates(update):
    print(f"Update: {update.context.id}")

await subscriber.start()
```

## Architecture

The SDK uses a client-server architecture:

- **ContextStore Client**: HTTP client for connecting to the Context Store server
- **FastAPI Server**: REST API server with WebSocket support and persistent storage
- **SQLite Backend**: Reliable persistent storage with semantic search capabilities
- **Subscription System**: Real-time updates via WebSocket connections

## Error Handling

```python
from episodic import ContextNotFoundException, ContextStoreException

try:
    context = await client.get("non.existent.id")
except ContextNotFoundException as e:
    print(f"Context not found: {e}")
except ContextStoreException as e:
    print(f"Store error: {e}")
```

## Acknowledgement
This work is done with the [Agentica](https://agentica-project.com/index.html) Team as part of [Berkeley Sky Computing Lab](https://sky.cs.berkeley.edu/). 