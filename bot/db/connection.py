"""Database connection and initialization."""

import os
import threading
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from bot.db.models import Base

# Database URL - configurable via environment
DATABASE_URL = os.environ.get(
    "PORTAL_DB",
    "sqlite+aiosqlite:///data/portal.db"
)

# Async engine and session maker
_engine = None
_session_maker = None
_init_thread_id: Optional[int] = None


def get_database_url() -> str:
    """Get the database URL, creating directory if needed."""
    url = DATABASE_URL
    # For SQLite, ensure data directory exists
    if url.startswith("sqlite"):
        # Extract path from URL
        db_path = url.replace("sqlite+aiosqlite:///", "")
        if db_path and db_path != ":memory:":
            db_dir = os.path.dirname(db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
    return url


async def _dispose_engine_safe(engine):
    """Safely dispose of an engine, handling event loop issues."""
    if engine is None:
        return
    try:
        await engine.dispose()
    except RuntimeError:
        # Event loop already closed, ignore
        pass


def init_engine(force_reinit: bool = False):
    """
    Initialize the database engine.
    
    Args:
        force_reinit: If True, force reinitialization even if engine exists.
                     Use this when starting in a new thread/event loop.
    """
    global _engine, _session_maker, _init_thread_id
    
    current_thread_id = threading.current_thread().ident
    
    # Check if we need to reinitialize:
    # 1. Force reinit requested, OR
    # 2. Engine exists but was created in a different thread
    needs_reinit = force_reinit or (_engine is not None and _init_thread_id != current_thread_id)
    
    if needs_reinit and _engine is not None:
        # Schedule disposal in the original thread's event loop if possible
        # For simplicity, we just replace the engine (old one will be GC'd)
        pass
    
    if _engine is None:
        url = get_database_url()
        _engine = create_async_engine(
            url,
            echo=False,
            poolclass=NullPool,  # Use NullPool for SQLite
        )
        _session_maker = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        _init_thread_id = current_thread_id
    
    return _engine, _session_maker


async def init_db():
    """Initialize database tables."""
    # Reinitialize for current thread (web server's event loop)
    reinit_engine_for_current_thread()
    engine, _ = init_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_db():
    """Close database connections."""
    global _engine, _session_maker, _init_thread_id
    
    if _engine is not None:
        await _dispose_engine_safe(_engine)
        _engine = None
        _session_maker = None
        _init_thread_id = None


def reinit_engine_for_current_thread():
    """
    Reinitialize the database engine for the current thread.
    
    Call this when entering a new thread/event loop context (e.g., when
    the web server starts in a separate thread from the Discord bot).
    This ensures database connections are bound to the correct event loop.
    """
    global _engine, _session_maker, _init_thread_id
    
    current_thread_id = threading.current_thread().ident
    
    # If already initialized for this thread, nothing to do
    if _engine is not None and _init_thread_id == current_thread_id:
        return
    
    # Create new engine for current thread
    url = get_database_url()
    _engine = create_async_engine(
        url,
        echo=False,
        poolclass=NullPool,
    )
    _session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    _init_thread_id = current_thread_id


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session context manager."""
    # Ensure engine is initialized for current thread
    reinit_engine_for_current_thread()
    _, session_maker = init_engine()
    session = session_maker()
    
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        try:
            await session.close()
        except Exception:
            # Ignore errors during close - session may be in bad state
            pass


async def get_db_session() -> AsyncSession:
    """Get a database session (caller must close)."""
    reinit_engine_for_current_thread()
    _, session_maker = init_engine()
    return session_maker()


# FastAPI dependency - yields a session and handles commit/rollback
async def get_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    This properly handles the async context manager protocol for FastAPI's Depends().
    Usage: async def endpoint(db: AsyncSession = Depends(get_db_dependency)):
    """
    _, session_maker = init_engine()
    
    async with session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()