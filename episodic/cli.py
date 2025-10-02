"""
Command Line Interface for Episodic Context Store.
Provides commands for storing, retrieving, and managing contexts.
"""

import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional, List
import argparse
import os

from .client import Episodic
from .store import SqliteContextStore
from .core import Context, ContextFilter


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        prog="episodic",
        description="Episodic Context Store CLI - Store and retrieve context data"
    )
    
    # Global options
    parser.add_argument(
        "--endpoint",
        default=os.getenv("EPISODIC_ENDPOINT", "sqlite://"),
        help="Context store endpoint: https://server.com for remote, sqlite:// for local (default: sqlite://)"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("EPISODIC_API_KEY", ""),
        help="API key for authentication"
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("EPISODIC_NAMESPACE", "default"),
        help="Default namespace (default: default)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Store command
    store_parser = subparsers.add_parser("store", help="Store a context")
    store_parser.add_argument("context_id", help="Unique context identifier")
    store_parser.add_argument("data", help="Context data as JSON string or @filename")
    store_parser.add_argument("--text", help="Optional text representation")
    store_parser.add_argument("--ttl", type=int, help="Time to live in seconds")
    store_parser.add_argument("--tags", nargs="*", help="List of tags")
    store_parser.add_argument("--type", default="generic", help="Context type")
    store_parser.add_argument("--metadata", help="Metadata as JSON string")
    store_parser.add_argument("--namespace-override", help="Override default namespace")
    
    # Get command
    get_parser = subparsers.add_parser("get", help="Get a context by ID")
    get_parser.add_argument("context_id", help="Context identifier")
    get_parser.add_argument("--format", choices=["json", "text", "table"], default="json", help="Output format")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query contexts with filters")
    query_parser.add_argument("--namespaces", nargs="*", help="Filter by namespaces")
    query_parser.add_argument("--tags", nargs="*", help="Filter by tags")
    query_parser.add_argument("--types", nargs="*", help="Filter by context types")
    query_parser.add_argument("--since", help="Filter by time (e.g., 1h, 30m, 1d)")
    query_parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    query_parser.add_argument("--format", choices=["json", "text", "table"], default="json", help="Output format")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search contexts")
    search_subparsers = search_parser.add_subparsers(dest="search_type", help="Search type")
    
    # Text search
    text_search_parser = search_subparsers.add_parser("text", help="Text search")
    text_search_parser.add_argument("query", help="Search query")
    text_search_parser.add_argument("--namespaces", nargs="*", help="Filter by namespaces")
    text_search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    text_search_parser.add_argument("--format", choices=["json", "text", "table"], default="json", help="Output format")
    
    # Semantic search
    semantic_search_parser = search_subparsers.add_parser("semantic", help="Semantic search")
    semantic_search_parser.add_argument("query", help="Search query")
    semantic_search_parser.add_argument("--namespaces", nargs="*", help="Filter by namespaces")
    semantic_search_parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")
    semantic_search_parser.add_argument("--limit", type=int, default=10, help="Maximum results")
    semantic_search_parser.add_argument("--format", choices=["json", "text", "table"], default="json", help="Output format")
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a context")
    delete_parser.add_argument("context_id", help="Context identifier")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run Episodic server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve_parser.add_argument("--db-path", help="SQLite database path")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    # Health command
    health_parser = subparsers.add_parser("health", help="Check context store health")
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="Start web dashboard")
    dashboard_parser.add_argument("--port", "-p", type=int, default=3000, help="Dashboard port (default: 3000)")
    dashboard_parser.add_argument("--context-store-port", "-c", type=int, default=8000, help="Context store port (default: 8000)")
    dashboard_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    return parser


def load_data_from_input(data_input: str) -> Dict[str, Any]:
    """Load data from JSON string or file."""
    if data_input.startswith("@"):
        # Load from file
        filename = data_input[1:]
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found", file=sys.stderr)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in file '{filename}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Parse as JSON string
        try:
            return json.loads(data_input)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON string: {e}", file=sys.stderr)
            sys.exit(1)


def format_output(data: Any, format_type: str) -> str:
    """Format output based on the specified format."""
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    elif format_type == "text":
        if isinstance(data, list):
            return "\n".join(str(item) for item in data)
        else:
            return str(data)
    elif format_type == "table":
        if isinstance(data, list) and data:
            # Simple table format for lists of contexts
            if isinstance(data[0], dict) and "id" in data[0]:
                lines = ["ID\tNamespace\tType\tCreated"]
                lines.append("-" * 50)
                for item in data:
                    created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item.get("created_at", 0)))
                    lines.append(f"{item.get('id', '')}\t{item.get('namespace', '')}\t{item.get('type', '')}\t{created}")
                return "\n".join(lines)
        return str(data)
    else:
        return str(data)


async def cmd_store(args, context_store) -> None:
    """Handle store command."""
    data = load_data_from_input(args.data)
    
    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid metadata JSON: {e}", file=sys.stderr)
            sys.exit(1)
    
    namespace = args.namespace_override or args.namespace
    
    try:
        context = await context_store.store(
            context_id=args.context_id,
            data=data,
            text=args.text,
            ttl=args.ttl,
            tags=args.tags or [],
            namespace=namespace,
            context_type=args.type,
            metadata=metadata
        )
        print(f"Stored context '{context.id}' in namespace '{context.namespace}'")
    except Exception as e:
        print(f"Error storing context: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_get(args, context_store) -> None:
    """Handle get command."""
    try:
        context = await context_store.get(args.context_id)
        output = format_output(context.to_dict(), args.format)
        print(output)
    except Exception as e:
        print(f"Error getting context: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_query(args, context_store) -> None:
    """Handle query command."""
    try:
        filter_obj = ContextFilter(
            namespaces=args.namespaces or [],
            tags=args.tags or [],
            context_types=args.types or [],
            since=args.since,
            limit=args.limit
        )
        contexts = await context_store.query(filter_obj)
        output = format_output([ctx.to_dict() for ctx in contexts], args.format)
        print(output)
    except Exception as e:
        print(f"Error querying contexts: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_search_text(args, context_store) -> None:
    """Handle text search command."""
    try:
        contexts = await context_store.search_text(
            query=args.query,
            namespaces=args.namespaces,
            limit=args.limit
        )
        output = format_output([ctx.to_dict() for ctx in contexts], args.format)
        print(output)
    except Exception as e:
        print(f"Error searching contexts: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_search_semantic(args, context_store) -> None:
    """Handle semantic search command."""
    try:
        contexts = await context_store.search_semantic(
            query=args.query,
            namespaces=args.namespaces,
            similarity_threshold=args.threshold,
            limit=args.limit
        )
        output = format_output([ctx.to_dict() for ctx in contexts], args.format)
        print(output)
    except Exception as e:
        print(f"Error searching contexts: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_delete(args, context_store) -> None:
    """Handle delete command."""
    try:
        success = await context_store.delete(args.context_id)
        if success:
            print(f"Deleted context '{args.context_id}'")
        else:
            print(f"Context '{args.context_id}' not found")
            sys.exit(1)
    except Exception as e:
        print(f"Error deleting context: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_health(args, context_store) -> None:
    """Handle health command."""
    try:
        health = await context_store.health_check()
        output = format_output(health, "json")
        print(output)
    except Exception as e:
        print(f"Error checking health: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_server(args) -> None:
    """Handle server command."""
    try:
        import uvicorn
        from .server import create_app
        
        print(f"Starting Episodic server on {args.host}:{args.port}")
        if args.db_path:
            print(f"Using database: {args.db_path}")
        
        # Create the app factory function for uvicorn
        def create_server_app():
            return create_app(db_path=args.db_path, namespace=args.namespace)
        
        # Run the server
        if args.reload:
            # For reload mode, pass the module path
            uvicorn.run(
                "episodic.server.app:create_app",
                host=args.host,
                port=args.port,
                reload=True,
                factory=True
            )
        else:
            # For normal mode, create app directly and use async server
            app = create_server_app()
            config = uvicorn.Config(
                app=app,
                host=args.host,
                port=args.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
    except ImportError:
        print("Error: uvicorn is required to run the server. Install with: pip install uvicorn", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


async def cmd_dashboard(args, context_store) -> None:
    """Handle dashboard command."""
    try:
        from .dashboard import run_dashboard_cli
        
        print("🧠 Starting Episodic Dashboard")
        
        # Run the dashboard (this will block until stopped)
        await run_dashboard_cli(args)
        
    except ImportError as e:
        print(f"Error: Dashboard dependencies not available: {e}", file=sys.stderr)
        print("Install with: pip install fastapi uvicorn", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error starting dashboard: {e}", file=sys.stderr)
        sys.exit(1)


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Handle serve command separately (doesn't need context store)
    if args.command == "serve":
        await cmd_server(args)
        return
    
    # Create context store based on endpoint
    if args.endpoint.startswith(('http://', 'https://')):
        # Remote endpoint - use Episodic client
        context_store = Episodic(
            endpoint=args.endpoint,
            api_key=args.api_key,
            namespace=args.namespace,
            timeout=args.timeout
        )
    else:
        # Local endpoint - use SqliteContextStore
        db_path = None if args.endpoint == "sqlite://" else args.endpoint.replace("sqlite://", "")
        context_store = SqliteContextStore(
            endpoint=args.endpoint,
            api_key=args.api_key,
            namespace=args.namespace,
            db_path=db_path
        )
    
    try:
        # Route to appropriate command handler
        if args.command == "store":
            await cmd_store(args, context_store)
        elif args.command == "get":
            await cmd_get(args, context_store)
        elif args.command == "query":
            await cmd_query(args, context_store)
        elif args.command == "search":
            if args.search_type == "text":
                await cmd_search_text(args, context_store)
            elif args.search_type == "semantic":
                await cmd_search_semantic(args, context_store)
            else:
                print("Error: Please specify search type (text or semantic)", file=sys.stderr)
                sys.exit(1)
        elif args.command == "delete":
            await cmd_delete(args, context_store)
        elif args.command == "health":
            await cmd_health(args, context_store)
        elif args.command == "dashboard":
            await cmd_dashboard(args, context_store)
        else:
            print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
            sys.exit(1)
    
    finally:
        # Clean up
        if hasattr(context_store, 'close'):
            await context_store.close()


def cli_main() -> None:
    """Entry point for the CLI script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    cli_main() 