"""FastAPI web server for llmcord portal."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.connection import close_db, init_db, get_db_dependency
from bot.web.config import get_portal_config

# Use "api" domain logger for HTTP request/response logging
logger = logging.getLogger("discord-bot.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan events for the web server."""
    # Startup
    logger.info("Starting web portal server...")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Check if portal is enabled
    config = get_portal_config()
    if not config.enabled:
        logger.warning("Portal is disabled in config")
    else:
        logger.info(f"Portal enabled on port {config.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down web portal server...")
    await close_db()


# Create FastAPI app
app = FastAPI(
    title="GPT Discord Bot",
    description="Web administration portal for llmcord Discord bot",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
)

# Conditionally add docs routes after app creation
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi import APIRouter

docs_router = APIRouter()

def is_docs_enabled() -> bool:
    """Check if docs are enabled at runtime (not just startup)."""
    try:
        config = get_portal_config()
        return config.docs_enabled
    except Exception:
        return False


@docs_router.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    if not is_docs_enabled():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
    )

@docs_router.get("/redoc", include_in_schema=False)
async def custom_redoc_html():
    if not is_docs_enabled():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
    )

@docs_router.get("/openapi.json", include_in_schema=False)
async def get_openapi_json():
    """OpenAPI schema endpoint - gated by docs_enabled."""
    if not is_docs_enabled():
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    return custom_openapi()

# Include docs router - runtime check happens in each endpoint
app.include_router(docs_router, tags=["Docs"])
logger.info("API documentation routes registered at /docs and /redoc (runtime gating enabled)")

# Define OpenAPI tags for endpoint grouping
openapi_tags = [
    {"name": "Health", "description": "Health check endpoints"},
    {"name": "Auth", "description": "Authentication and user management"},
    {"name": "Status", "description": "Bot status and presence"},
    {"name": "Config", "description": "Configuration management"},
    {"name": "Tasks", "description": "Task management"},
    {"name": "Skills", "description": "Skill management"},
    {"name": "Servers", "description": "Server management"},
    {"name": "Personas", "description": "Persona management"},
    {"name": "Logs", "description": "Log management"},
    {"name": "WebSocket", "description": "Real-time WebSocket endpoints"},
]

# OAuth2 security scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def custom_openapi():
    """Generate custom OpenAPI schema with security scheme."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add tags
    openapi_schema["tags"] = openapi_tags
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "description": "Enter your JWT token (no prefix needed)",
        }
    }
    
    # Apply security to all endpoints except health, setup, and has-users
    for path, path_item in openapi_schema.get("paths", {}).items():
        for method, operation in path_item.items():
            if method in ["get", "post", "put", "delete", "patch"]:
                # Skip public endpoints
                if path in ["/health", "/openapi.json", "/docs", "/redoc"]:
                    continue
                if path == "/api/auth/setup":
                    continue
                if path == "/api/auth/has-users":
                    continue
                # Add security requirement
                operation["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Override the default OpenAPI endpoint
app.openapi = custom_openapi

# Add CORS middleware
config = get_portal_config()
cors_origins = config.cors_origins if config.cors_origins else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,  # Empty = same origin only (secure)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_web_server() -> None:
    """Run the web server (blocking)."""
    import uvicorn
    import logging
    import os
    from datetime import datetime, timezone, timedelta
    import json
    import warnings

    # Suppress Python warnings in the web server
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    # Get local timezone offset (cached for performance)
    _cached_tz_offset: str | None = None

    def get_local_timezone_offset() -> str:
        """Get local timezone offset as ISO 8601 string (e.g., +08:00)."""
        nonlocal _cached_tz_offset
        if _cached_tz_offset is None:
            now = datetime.now()
            utc_offset = now.astimezone().utcoffset()
            if utc_offset is None:
                _cached_tz_offset = "Z"
            else:
                total_seconds = int(utc_offset.total_seconds())
                hours, remainder = divmod(abs(total_seconds), 3600)
                minutes = remainder // 60
                sign = "+" if total_seconds >= 0 else "-"
                _cached_tz_offset = f"{sign}{hours:02d}:{minutes:02d}"
        return _cached_tz_offset

    config = get_portal_config()
    if not config.enabled:
        logger.info("Portal disabled, skipping web server start")
        return

    # Get environment for log formatting
    environment = os.environ.get("ENVIRONMENT", "development")
    service = os.environ.get("LOG_SERVICE", "gpt-discord-bot-portal")

    # Create StructuredFormatter for Uvicorn logs
    class UvicornStructuredFormatter(logging.Formatter):
        """Structured formatter for Uvicorn logs matching our app format."""

        def __init__(self, service: str, environment: str):
            super().__init__()
            self.service = service
            self.environment = environment

        def format(self, record: logging.LogRecord) -> str:
            # Use timezone-aware UTC datetime for consistent timestamps
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + get_local_timezone_offset(),
                "level": record.levelname,
                "message": record.getMessage(),
                "service": self.service,
                "environment": self.environment,
                "source": {
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno,
                },
            }

            # Add error context for ERROR and CRITICAL
            if record.levelno >= logging.ERROR and record.exc_info:
                log_data.update({
                    "error_type": record.exc_info[0].__name__ if record.exc_info[0] else "Exception",
                    "error_message": str(record.exc_info[1]) if record.exc_info[1] else "",
                    "stack": self.formatException(record.exc_info),
                })

            if self.environment == "production":
                return json.dumps(log_data)
            else:
                # Human-readable format
                return f"{log_data['timestamp']} [{log_data['level']}] {self.service} | {record.getMessage()}"

    # Configure Uvicorn loggers to use structured formatting
    formatter = UvicornStructuredFormatter(service=service, environment=environment)

    # Override uvicorn default logger
    uvicorn_default = logging.getLogger("uvicorn.default")
    uvicorn_default.handlers.clear()
    uvicorn_default.setLevel(logging.INFO)
    uvicorn_default.addHandler(logging.StreamHandler())
    for handler in uvicorn_default.handlers:
        handler.setFormatter(formatter)

    # Override uvicorn access logger
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers.clear()
    uvicorn_access.setLevel(logging.INFO)
    uvicorn_access.addHandler(logging.StreamHandler())
    for handler in uvicorn_access.handlers:
        handler.setFormatter(formatter)

    # Override uvicorn ASGI logger
    uvicorn_asgi = logging.getLogger("uvicorn.asgi")
    uvicorn_asgi.handlers.clear()
    uvicorn_asgi.setLevel(logging.INFO)
    uvicorn_asgi.addHandler(logging.StreamHandler())
    for handler in uvicorn_asgi.handlers:
        handler.setFormatter(formatter)

    logger.info(f"Starting web server on port {config.port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.port,
        log_level="info",
    )


# Import routes after app creation to avoid circular imports
from bot.web.auth import (
    setup_portal,
    login,
    get_users,
    check_has_users,
    SetupRequest,
    LoginRequest,
    Token,
    CurrentUser,
    get_current_user,
)


# Health check endpoint for container orchestration
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Docker/container health checks.
    
    Returns the current health status of the web portal service.
    """
    return {"status": "healthy", "service": "gpt-discord-bot-portal"}


# Request logging middleware (21.1.2)
from fastapi import Request
import time


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with timing information."""
    # Skip logging for /api/logs and /ws/logs endpoints to prevent feedback loop in LogViewer
    if request.url.path.startswith("/api/logs") or request.url.path.startswith("/ws/logs"):
        return await call_next(request)
    
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Log the request
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} duration={duration:.3f}s"
    )
    
    return response


# Auth routes
@app.post("/api/auth/setup", response_model=Token, tags=["Auth"])
async def api_setup(request: SetupRequest, db: AsyncSession = Depends(get_db_dependency)):
    """
    First-time setup - create initial admin user.
    
    Use this endpoint only when no users exist in the system yet.
    Creates the first admin user with the provided credentials.
    """
    return await setup_portal(request, db)


@app.get("/api/auth/has-users", tags=["Auth"])
async def api_has_users(db: AsyncSession = Depends(get_db_dependency)):
    """
    Check if any users exist in the system.
    
    Use this to determine whether to show the setup wizard or login page.
    """
    has_users = await check_has_users(db)
    return {"has_users": has_users}


@app.post("/api/auth/login", response_model=Token, tags=["Auth"])
async def api_login(request: LoginRequest, db: AsyncSession = Depends(get_db_dependency)):
    """
    Login with username and password.
    
    Returns a JWT token that should be used for subsequent authenticated requests.
    """
    return await login(request, db)


@app.get("/api/auth/me", tags=["Auth"])
async def api_me(current_user: CurrentUser = Depends(get_current_user)):
    """
    Verify the current token and get user info.
    
    Use this to verify a token is still valid after database refresh.
    Returns user info if token is valid, 401 otherwise.
    """
    return {"id": current_user.id, "username": current_user.username}


@app.get("/api/auth/users", response_model=list[dict], tags=["Auth"])
async def api_get_users(
    db: AsyncSession = Depends(get_db_dependency),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Get list of all users.
    
    Requires authentication. Returns basic user information (passwords are not included).
    """
    return await get_users(db, current_user)


# Status & Servers routes
from bot.web.routes.status import router as status_router
from bot.web.routes.servers import router as servers_router
from bot.web.routes.logs import router as logs_router
from bot.web.routes.config import router as config_router
from bot.web.routes.personas import router as personas_router
from bot.web.routes.tasks import router as tasks_router
from bot.web.routes.skills import router as skills_router
from bot.web.routes.tools import router as tools_router

app.include_router(status_router, tags=["Status"])
app.include_router(servers_router, tags=["Servers"])
app.include_router(logs_router, tags=["Logs"])
app.include_router(config_router, tags=["Config"])
app.include_router(personas_router, tags=["Personas"])
app.include_router(tasks_router, tags=["Tasks"])
app.include_router(skills_router, tags=["Skills"])
app.include_router(tools_router, tags=["Tools"])


# WebSocket endpoint for real-time logs
from fastapi import WebSocket, WebSocketDisconnect
from bot.web.log_handler import add_log_client, remove_log_client, get_log_client_count


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """
    WebSocket endpoint for real-time log streaming.
    
    Clients connect to receive log events as they occur.
    The logs are broadcast from the DatabaseLogHandler.
    
    **Connection:**
    - Connect with: `ws://host/ws/logs?token=YOUR_JWT_TOKEN`
    - Send "ping" to check connection health
    - Receive JSON log entries with: level, timestamp, message
    """
    await websocket.accept()
    add_log_client(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {get_log_client_count()}")
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for any message from client (ping/pong mechanism)
            data = await websocket.receive_text()
            
            # Echo back for connection health check
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        remove_log_client(websocket)
        logger.info(f"WebSocket client removed. Total clients: {get_log_client_count()}")


@app.get("/ws/status")
async def websocket_status():
    """
    Get WebSocket connection status.
    
    Returns information about active WebSocket connections.
    """
    return {
        "log_clients": get_log_client_count(),
        "status": "connected" if get_log_client_count() > 0 else "no_clients",
    }


# Serve React frontend static files
# Determine the path to the web/dist folder
import pathlib

def get_frontend_dist_path() -> pathlib.Path:
    """Get the path to the frontend dist folder."""
    # Go up from bot/web/ to project root, then into web/dist
    project_root = pathlib.Path(__file__).parent.parent.parent
    return project_root / "web" / "dist"


frontend_dist = get_frontend_dist_path()

if frontend_dist.exists():
    # Mount all static files from dist root
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="static")
    
    # Catch-all route for SPA - only for non-API routes
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve SPA for any non-API route."""
        # Skip if it's an API route
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not Found")
        
        # Serve index.html for SPA routing
        from fastapi.responses import FileResponse
        index_path = frontend_dist / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return {"detail": "Not found"}
    
    logger.info(f"Serving React frontend from {frontend_dist}")
else:
    logger.warning(f"Frontend dist folder not found at {frontend_dist}")