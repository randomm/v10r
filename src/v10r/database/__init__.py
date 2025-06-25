"""
Database integration package for v10r.

This package provides:
- Async PostgreSQL connection pooling
- Database schema introspection
- Column existence and type checking
- Database health checks
"""

from .connection import DatabaseManager, ConnectionPool
from .introspection import SchemaIntrospector, ColumnInfo, TableInfo
from .health import DatabaseHealthChecker

__all__ = [
    "DatabaseManager",
    "ConnectionPool",
    "SchemaIntrospector", 
    "ColumnInfo",
    "TableInfo",
    "DatabaseHealthChecker",
] 