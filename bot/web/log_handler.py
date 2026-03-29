"""Log handler for dual logging (console + database)."""

import logging
import asyncio
import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.connection import get_db
from bot.db.models import EventLog
from bot.web.config import get_portal_config

logger = logging.getLogger(__name__)

# Global list to track connected WebSocket clients for real-time logs
_log_clients: list = []


def _get_local_timezone_offset() -> str:
    """Get local timezone offset as ISO 8601 string (e.g., +08:00)."""
    now = datetime.now()
    utc_offset = now.astimezone().utcoffset()
    if utc_offset is None:
        return "Z"
    total_seconds = int(utc_offset.total_seconds())
    hours, remainder = divmod(abs(total_seconds), 3600)
    minutes = remainder // 60
    sign = "+" if total_seconds >= 0 else "-"
    return f"{sign}{hours:02d}:{minutes:02d}"


class DatabaseLogHandler(logging.Handler):
    """
    Custom logging handler that writes logs to both console and database.
    
    This handler:
    1. Emits logs to the standard console handler (preserving normal logging)
    2. Writes logs to the database for the web portal's log viewer
    3. Broadcasts logs to connected WebSocket clients
    """
    
    def __init__(self):
        super().__init__()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._pending_logs: list = []
        self._batch_size = 10
        self._batch_timeout = 2.0  # seconds
        
        # Read sampling rate from environment - do this early!
        # This ensures sampling works even if set_loop is never called
        sampling_rate = os.environ.get("LOG_SAMPLING_RATE", "1.0")
        try:
            self._sampling_rate = float(sampling_rate)
        except ValueError:
            self._sampling_rate = 1.0
    
    def filter(self, record: logging.LogRecord) -> logging.LogRecord | None:
        """
        First-line filter to block SQLAlchemy/asyncio DEBUG noise BEFORE emit is called.
        This prevents catastrophic I/O from database connection debug messages.
        """
        # Block ALL DEBUG logs that contain SQLAlchemy keywords
        # These logs have our logger name "discord-bot" but contain SQLAlchemy messages
        if record.levelno == logging.DEBUG:
            try:
                msg = record.getMessage()
                # These keywords indicate SQLAlchemy internal operations
                if any(kw in msg for kw in (
                    "executing", "operation ", "connect.<locals>",
                    "create_function", "regexp", "sqlite3"
                )):
                    return None  # Drop the record
            except Exception:
                pass
        
        return record  # Allow all other logs
    
    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the asyncio event loop for async operations."""
        self._loop = loop
        # Update sampling rate from environment
        sampling_rate = os.environ.get("LOG_SAMPLING_RATE", "1.0")
        try:
            self._sampling_rate = float(sampling_rate)
        except ValueError:
            self._sampling_rate = 1.0
    
    def _should_sample(self, level: str) -> bool:
        """
        Determine if this log should be sampled based on sampling rate.
        Uses true random sampling per log entry in production mode.
        """
        import random
        
        # Always log errors regardless of sampling
        if level in ("ERROR", "CRITICAL"):
            return True
        
        # In development, always log
        environment = os.environ.get("ENVIRONMENT", "development")
        if environment != "production":
            return True
        
        # Apply sampling rate - true random per log (not cached)
        return random.random() < self._sampling_rate
    
    def emit(self, record: logging.LogRecord):
        """Emit a log record to console and schedule database write."""
        try:
            # ULTRA AGGRESSIVE: Skip ANY SQLAlchemy/asyncio DEBUG logs at the door
            # This prevents catastrophic I/O from database connection flood
            if record.levelno == logging.DEBUG:
                # Check logger name
                if record.name and (
                    record.name.startswith("sqlalchemy.") or 
                    record.name.startswith("asyncio.")
                ):
                    # Check message content for SQLAlchemy operations
                    msg = record.getMessage()
                    if any(keyword in msg for keyword in (
                        "executing", "operation", "connect", "connection", 
                        "create_function", "regexp", "sqlite3"
                    )):
                        return
            
            # Check sampling rate (skip some logs in production if configured)
            if not self._should_sample(record.levelname):
                return
            
            # Get log level from record
            level = record.levelname
            
            # Skip SQLAlchemy internal logs to prevent log flood
            if record.name and (record.name.startswith("sqlalchemy.") or record.name.startswith("asyncio.")):
                return
            
            # Check if we should log this level (based on config)
            try:
                config = get_portal_config()
                allowed_levels = config.logs_levels
            except Exception:
                # If config fails, allow all
                allowed_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            
            if level not in allowed_levels:
                return
            
            # Get structured fields from record (following logging-guide skill)
            service = os.environ.get("LOG_SERVICE", "discord-bot")
            environment = os.environ.get("ENVIRONMENT", "development")
            
            # Create log entry data with ISO 8601 timestamp in local timezone
            log_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + _get_local_timezone_offset(),
                "level": level,
                "event_type": record.name or "unknown",
                "message": record.getMessage(),
                "extra_data": {
                    "service": service,
                    "environment": environment,
                    "source": {
                        "module": record.module,
                        "function": record.funcName,
                        "line": record.lineno,
                    },
                },
            }
            
            # Add recommended fields (trace_id, request_id, user_id) if available
            for field_name in ("trace_id", "span_id", "user_id", "request_id"):
                if hasattr(record, field_name) and getattr(record, field_name):
                    log_data["extra_data"][field_name] = getattr(record, field_name)
            
            # Add error context for ERROR and CRITICAL levels (per logging-guide)
            if record.levelno >= logging.ERROR and record.exc_info:
                log_data["extra_data"]["error_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "Exception"
                log_data["extra_data"]["error_message"] = str(record.exc_info[1]) if record.exc_info[1] else ""
                log_data["extra_data"]["stack"] = self.formatException(record.exc_info)
            
            # Schedule async write
            if self._loop and self._loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self._write_log_async(log_data), 
                    self._loop
                )
            else:
                # Store for later if loop not available
                self._pending_logs.append(log_data)
                
        except Exception as e:
            # Don't let logging errors crash the app
            print(f"Error in log handler: {e}")
    
    async def _write_log_async(self, log_data: dict):
        """Write log to database asynchronously."""
        try:
            # Parse ISO 8601 timestamp string back to datetime
            timestamp_str = log_data["timestamp"]
            # Remove Z suffix and microseconds for parsing
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1]
            timestamp_dt = datetime.fromisoformat(timestamp_str)
            
            try:
                async with get_db() as session:
                    event_log = EventLog(
                        timestamp=timestamp_dt,
                        level=log_data["level"],
                        event_type=log_data["event_type"],
                        message=log_data["message"],
                        extra_data=log_data.get("extra_data"),
                    )
                    session.add(event_log)
                    await session.commit()
            except Exception as db_error:
                # Handle concurrent database access errors gracefully
                # (SQLite doesn't handle concurrent writes well)
                logger.warning(f"DB write error (will skip persistence): {db_error}")
                # Still broadcast even if DB write fails
            
            # Broadcast to WebSocket clients
            await self._broadcast_log(log_data)
            
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    async def _broadcast_log(self, log_data: dict):
        """Broadcast log to all connected WebSocket clients."""
        if not _log_clients:
            return
        
        # Format log for JSON broadcast (include structured fields per logging-guide)
        broadcast_data = {
            "timestamp": log_data["timestamp"],  # Already ISO 8601 string
            "level": log_data["level"],
            "event_type": log_data["event_type"],
            "logger": log_data["event_type"],  # Add logger field for LogViewer compatibility
            "message": log_data["message"],
        }
        
        # Add structured fields from extra_data
        if log_data.get("extra_data"):
            extra = log_data["extra_data"]
            if "service" in extra:
                broadcast_data["service"] = extra["service"]
            if "environment" in extra:
                broadcast_data["environment"] = extra["environment"]
            # Include trace_id, request_id, user_id if present
            for field in ("trace_id", "request_id", "user_id"):
                if field in extra:
                    broadcast_data[field] = extra[field]
        
        # Remove clients that have closed
        dead_clients = []
        for client in _log_clients:
            try:
                await client.send_json(broadcast_data)
            except Exception:
                dead_clients.append(client)
        
        # Clean up dead clients
        for client in dead_clients:
            if client in _log_clients:
                _log_clients.remove(client)


def add_log_client(client):
    """Add a WebSocket client to receive real-time logs."""
    if client not in _log_clients:
        _log_clients.append(client)
        logger.info(f"WebSocket client added. Total clients: {len(_log_clients)}")


def remove_log_client(client):
    """Remove a WebSocket client from receiving logs."""
    if client in _log_clients:
        _log_clients.remove(client)
        logger.info(f"WebSocket client removed. Total clients: {len(_log_clients)}")


def get_log_client_count() -> int:
    """Get the number of connected log clients."""
    return len(_log_clients)


async def cleanup_old_logs(retention_days: int = 7):
    """
    Clean up logs older than retention_days.
    
    This should be called periodically (e.g., daily) to prevent database bloat.
    """
    try:
        async with get_db() as session:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            # Delete old logs
            result = await session.execute(
                select(EventLog.id).where(EventLog.timestamp < cutoff_date)
            )
            old_log_ids = result.scalars().all()
            
            if old_log_ids:
                await session.execute(
                    EventLog.__table__.delete().where(EventLog.id.in_(old_log_ids))
                )
                await session.commit()
                logger.info(f"Cleaned up {len(old_log_ids)} old log entries")
                
    except Exception as e:
        logger.error(f"Failed to cleanup old logs: {e}")


def setup_log_handler() -> DatabaseLogHandler:
    """
    Set up and return the database log handler.
    
    Call this during bot startup to enable dual logging.
    """
    handler = DatabaseLogHandler()
    
    # Set the asyncio loop
    try:
        loop = asyncio.get_event_loop()
        handler.set_loop(loop)
    except RuntimeError:
        # No event loop yet, will set later
        pass
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    
    logger.info("Database log handler initialized")
    
    return handler


# Module-level function to initialize logging during bot startup
def init_logging():
    """Initialize the logging system with database handler."""
    return setup_log_handler()