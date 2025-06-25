"""
Exception classes for v10r.
"""

from typing import Any, Dict, Optional


class V10rError(Exception):
    """Base exception for all v10r errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause

    def __str__(self) -> str:
        result = self.message
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            result += f" [{details_str}]"
        if self.cause:
            result += f" (caused by: {self.cause})"
        return result


class ConfigurationError(V10rError):
    """Raised when there's an error in configuration."""

    pass


class ValidationError(V10rError):
    """Raised when there's a validation error."""

    pass


class DatabaseError(V10rError):
    """Raised when there's an error with database operations."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when there's an error establishing or maintaining database connections."""

    pass


class DatabaseConfigurationError(DatabaseError):
    """Raised when there's an error in database configuration."""

    pass


class SchemaError(DatabaseError):
    """Raised when there's an error with database schema operations."""

    pass


class ColumnCollisionError(SchemaError):
    """Raised when there's a critical column name collision."""

    def __init__(
        self,
        table_name: str,
        column_name: str,
        collision_type: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Critical column collision in table '{table_name}': "
            f"column '{column_name}' ({collision_type})",
            details,
        )
        self.table_name = table_name
        self.column_name = column_name
        self.collision_type = collision_type


class DimensionMismatchError(SchemaError):
    """Raised when vector dimensions don't match expected dimensions."""

    def __init__(
        self,
        expected_dim: int,
        actual_dim: int,
        context: Optional[str] = None,
    ) -> None:
        message = f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
        if context:
            message += f" ({context})"
        super().__init__(message)
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim


class EmbeddingError(V10rError):
    """Raised when there's an error with embedding operations."""

    pass


class EmbeddingAPIError(EmbeddingError):
    """Raised when there's an error with embedding API calls."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        api_provider: Optional[str] = None,
    ) -> None:
        details = {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body
        if api_provider:
            details["api_provider"] = api_provider

        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body
        self.api_provider = api_provider


class APITimeoutError(EmbeddingAPIError):
    """Raised when API calls timeout."""

    def __init__(
        self,
        message: str = "API request timed out",
        timeout_duration: Optional[float] = None,
        api_provider: Optional[str] = None,
    ) -> None:
        if timeout_duration:
            message += f" (timeout: {timeout_duration}s)"
        
        super().__init__(message, api_provider=api_provider)
        self.timeout_duration = timeout_duration


class RateLimitError(EmbeddingAPIError):
    """Raised when hitting API rate limits."""

    def __init__(
        self,
        retry_after: Optional[int] = None,
        api_provider: Optional[str] = None,
    ) -> None:
        message = "API rate limit exceeded"
        if retry_after:
            message += f", retry after {retry_after} seconds"

        details = {}
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(message, api_provider=api_provider)
        self.retry_after = retry_after


class InvalidTextError(EmbeddingError):
    """Raised when text cannot be processed for embedding."""

    def __init__(self, text: str, reason: str) -> None:
        super().__init__(f"Invalid text for embedding: {reason}")
        self.text = text
        self.reason = reason


class ProcessingError(V10rError):
    """Raised when there's an error in text processing."""

    pass


class CleaningError(ProcessingError):
    """Raised when there's an error in text cleaning."""

    pass


class QueueError(V10rError):
    """Raised when there's an error with queue operations."""

    pass


class ListenerError(V10rError):
    """Raised when there's an error with the listener service."""

    pass


class WorkerError(V10rError):
    """Raised when there's an error with worker operations."""

    pass


class RegistrationError(V10rError):
    """Raised when there's an error with table registration operations."""

    pass 