#!/usr/bin/env python3
"""
Episodic Dashboard

HTTP server for the Episodic Context Store Dashboard
"""

import asyncio
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


def create_dashboard_app() -> FastAPI:
    """Create the dashboard FastAPI app"""
    
    app = FastAPI(
        title="Episodic Context Store Dashboard",
        description="Real-time web dashboard for monitoring context store data",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard_home():
        """Serve the dashboard HTML"""
        dashboard_file = Path(__file__).parent / "dashboard.html"
        
        if not dashboard_file.exists():
            return HTMLResponse(
                content="<h1>Dashboard not found</h1><p>dashboard.html file is missing</p>",
                status_code=404
            )
        
        # Read and return the dashboard HTML content
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)

    @app.get("/health")
    async def dashboard_health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "Episodic Context Store Dashboard",
            "timestamp": time.time()
        }

    @app.get("/config")
    async def dashboard_config():
        """Get current dashboard configuration"""
        return {
            "default_context_store_url": "http://localhost:8000",
            "default_websocket_url": "ws://localhost:8000",
            "dashboard_version": "1.0.0"
        }
    
    return app


async def start_dashboard_server(
    port: int = 3000,
    open_browser: bool = True,
    context_store_port: int = 8000
):
    """Start the dashboard HTTP server"""
    
    print(f"🌐 Starting Episodic Dashboard")
    print(f"📡 Dashboard: http://localhost:{port}")
    print(f"🔗 Context Store: http://localhost:{context_store_port}")
    print("=" * 50)
    
    # Create the app
    app = create_dashboard_app()
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False  # Reduce noise
    )
    
    server = uvicorn.Server(config)
    
    # Open browser after a short delay
    if open_browser:
        async def open_browser_delayed():
            await asyncio.sleep(2)
            dashboard_url = f"http://localhost:{port}"
            print(f"🚀 Opening dashboard: {dashboard_url}")
            webbrowser.open(dashboard_url)
        
        asyncio.create_task(open_browser_delayed())
    
    # Start server
    await server.serve()


async def run_dashboard_cli(args):
    """Run dashboard from CLI"""
    try:
        await start_dashboard_server(
            port=args.port,
            open_browser=not args.no_browser,
            context_store_port=args.context_store_port
        )
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # For direct execution
    import argparse
    
    parser = argparse.ArgumentParser(description="Start Episodic Dashboard")
    parser.add_argument("--port", "-p", type=int, default=3000, help="Dashboard port (default: 3000)")
    parser.add_argument("--context-store-port", "-c", type=int, default=8000, help="Context store port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    asyncio.run(run_dashboard_cli(args)) 