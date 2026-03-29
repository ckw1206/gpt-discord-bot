"""Logs API endpoints."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from bot.db.models import EventLog
from bot.db.connection import get_db_dependency
from bot.web.auth import get_current_user
from bot.web.config import get_portal_config


def _format_timestamp(dt: Optional[datetime]) -> str:
    """Format datetime as ISO 8601 with local timezone offset."""
    if dt is None:
        return ""
    # If datetime is naive (no timezone), assume it's local time
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
    return dt.isoformat()

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["logs"])


# Response models
class LogEntry(BaseModel):
    """Single log entry."""
    id: int
    timestamp: str
    level: str
    event_type: str
    logger: Optional[str] = None  # Alias for event_type (LogViewer compatibility)
    message: str
    metadata: Optional[dict] = None
    # Additional structured fields from extra_data (per logging-guide skill)
    service: Optional[str] = None
    environment: Optional[str] = None

    model_config = {"from_attributes": True}


class LogsResponse(BaseModel):
    """Response for logs query."""
    logs: list[LogEntry]
    total: int
    page: int
    page_size: int


# Query parameters model
class LogQueryParams:
    """Query parameters for log filtering."""
    
    def __init__(
        self,
        level: Optional[str] = Query(None, description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"),
        event_type: Optional[str] = Query(None, description="Filter by event type"),
        since: Optional[str] = Query(None, description="ISO format timestamp to filter logs since"),
        until: Optional[str] = Query(None, description="ISO format timestamp to filter logs until"),
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(50, ge=1, le=500, description="Items per page"),
    ):
        self.level = level
        self.event_type = event_type
        self.since = since
        self.until = until
        self.page = page
        self.page_size = page_size


@router.get("/logs", response_model=LogsResponse)
async def get_logs(
    db: AsyncSession = Depends(get_db_dependency),
    level: Optional[str] = Query(None, description="Filter by log level"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    logger: Optional[str] = Query(None, description="Filter by logger (alias for event_type)"),
    service: Optional[str] = Query(None, description="Filter by service name"),
    environment: Optional[str] = Query(None, description="Filter by environment"),
    since: Optional[str] = Query(None, description="ISO timestamp to filter logs since"),
    until: Optional[str] = Query(None, description="ISO timestamp to filter logs until"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=500, description="Items per page"),
) -> LogsResponse:
    """
    Get event logs with optional filtering.
    
    - **level**: Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - **event_type**: Filter by event type
    - **since**: ISO format timestamp - only return logs after this time
    - **until**: ISO format timestamp - only return logs before this time
    - **page**: Page number (1-indexed)
    - **page_size**: Number of items per page (max 500)
    """
    # Get config for allowed log levels
    config = get_portal_config()
    allowed_levels = config.logs_levels
    
    # Build filters
    filters = []
    
    # Level filter - only allow levels configured in portal
    if level:
        level_upper = level.upper()
        # If the requested level is in allowed_levels, filter by that level
        # If not in allowed_levels, return NO results (user requested disabled level)
        if level_upper in allowed_levels:
            filters.append(EventLog.level == level_upper)
        else:
            # Level not in allowed config - return no results
            filters.append(EventLog.level == "__NO_MATCH__")
    else:
        # If no level specified, show only allowed levels
        filters.append(EventLog.level.in_(allowed_levels))
    
    # Event type filter (also handle 'logger' alias)
    if event_type or logger:
        filter_value = event_type or logger
        filters.append(EventLog.event_type == filter_value)
    
    # Note: Service/environment filtering is done in Python after fetch
    # (to support both SQLite and PostgreSQL)
    filter_service = service
    filter_environment = environment
    
    # Time range filters
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace('Z', '+00:00'))
            filters.append(EventLog.timestamp >= since_dt)
        except ValueError:
            logger.warning(f"Invalid 'since' timestamp format: {since}")
    
    if until:
        try:
            until_dt = datetime.fromisoformat(until.replace('Z', '+00:00'))
            filters.append(EventLog.timestamp <= until_dt)
        except ValueError:
            logger.warning(f"Invalid 'until' timestamp format: {until}")
    
    # Build query with filters
    query = select(EventLog)
    if filters:
        query = query.where(and_(*filters))
    
    # Order by timestamp descending (newest first)
    query = query.order_by(EventLog.timestamp.desc())
    
    # Get total count
    from sqlalchemy import func
    if filters:
        count_result = await db.execute(
            select(func.count()).where(and_(*filters))
        )
    else:
        count_result = await db.execute(
            select(func.count()).select_from(EventLog)
        )
    total = count_result.scalar() or 0
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # Execute query
    result = await db.execute(query)
    logs = result.scalars().all()
    
    # Convert to response format (extract structured fields from extra_data)
    log_entries = []
    for log in logs:
        # Extract service and environment from extra_data JSON
        extra = log.extra_data or {}
        log_service = extra.get("service")
        log_environment = extra.get("environment")
        
        # Apply service/environment filters (Python-side for DB compatibility)
        if filter_service and log_service != filter_service:
            continue
        if filter_environment and log_environment != filter_environment:
            continue
        
        # Filter out service/environment from metadata display (they're top-level now)
        metadata = {k: v for k, v in extra.items() if k not in ("service", "environment")}
        if not metadata:
            metadata = None
        
        log_entries.append(LogEntry(
            id=log.id,
            timestamp=_format_timestamp(log.timestamp),
            level=log.level,
            event_type=log.event_type,
            logger=log.event_type,  # Alias for LogViewer compatibility
            message=log.message,
            metadata=metadata,
            service=log_service,
            environment=log_environment,
        ))
    
    return LogsResponse(
        logs=log_entries,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/logs/levels", response_model=list[str])
async def get_log_levels() -> list[str]:
    """Get available log levels based on portal configuration."""
    config = get_portal_config()
    return config.logs_levels


@router.get("/logs/types", response_model=list[str])
async def get_log_types(db: AsyncSession = Depends(get_db_dependency)) -> list[str]:
    """Get list of unique event types in the logs."""
    result = await db.execute(
        select(EventLog.event_type).distinct().order_by(EventLog.event_type)
    )
    types = result.scalars().all()
    return list(types)