"""Shared Pydantic schemas for the web portal API."""

from typing import Any
from pydantic import BaseModel, Field


class HTTPError(BaseModel):
    """Standard HTTP error response."""
    detail: str = Field(description="Error message")
    error_code: str | None = Field(default=None, description="Optional error code")


class NotFoundError(BaseModel):
    """Resource not found error response."""
    detail: str = Field(default="Resource not found")
    resource_type: str | None = Field(default=None, description="Type of resource that was not found")
    resource_id: str | None = Field(default=None, description="ID of the resource that was not found")


class ValidationError(BaseModel):
    """Validation error response."""
    detail: list[dict[str, Any]] = Field(description="List of validation errors")
    body: dict[str, Any] | None = Field(default=None, description="The invalid request body")


class SuccessResponse(BaseModel):
    """Standard success response."""
    success: bool = Field(default=True, description="Indicates success")
    message: str | None = Field(default=None, description="Optional success message")
    data: dict[str, Any] | None = Field(default=None, description="Optional response data")