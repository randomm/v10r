"""
Database schema introspection for v10r.

Provides utilities for examining PostgreSQL database schemas,
including column information, table metadata, and pgvector specifics.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Set, Tuple
from enum import Enum

import asyncpg
from pydantic import BaseModel

from .connection import ConnectionPool
from ..exceptions import DatabaseError, SchemaError


logger = logging.getLogger(__name__)


class ColumnType(str, Enum):
    """Supported column types for v10r operations."""
    
    TEXT = "text"
    VARCHAR = "varchar"
    VECTOR = "vector"
    INTEGER = "integer"
    BIGINT = "bigint"
    TIMESTAMP = "timestamp"
    TIMESTAMPTZ = "timestamptz"
    BOOLEAN = "boolean"
    JSON = "json"
    JSONB = "jsonb"
    UNKNOWN = "unknown"


@dataclass
class ColumnInfo:
    """Information about a database column."""
    
    name: str
    data_type: str
    is_nullable: bool
    default_value: Optional[str] = None
    default: Optional[str] = None  # Alias for tests
    max_length: Optional[int] = None
    numeric_precision: Optional[int] = None
    numeric_scale: Optional[int] = None
    
    # pgvector specific
    vector_dimension: Optional[int] = None
    
    # Additional metadata
    ordinal_position: int = 0
    udt_name: Optional[str] = None
    is_generated: bool = False
    generation_expression: Optional[str] = None
    
    def __post_init__(self):
        # Sync default_value and default for compatibility
        if self.default is None and self.default_value is not None:
            self.default = self.default_value
        elif self.default is not None and self.default_value is None:
            self.default_value = self.default
    
    @property
    def column_type(self) -> ColumnType:
        """Get the standardized column type."""
        if self.data_type.lower() in ("text", "character varying", "varchar"):
            return ColumnType.TEXT if self.data_type.lower() == "text" else ColumnType.VARCHAR
        elif self.udt_name == "vector":
            return ColumnType.VECTOR
        elif self.data_type.lower() in ("integer", "int4"):
            return ColumnType.INTEGER
        elif self.data_type.lower() in ("bigint", "int8"):
            return ColumnType.BIGINT
        elif self.data_type.lower() == "timestamp without time zone":
            return ColumnType.TIMESTAMP
        elif self.data_type.lower() == "timestamp with time zone":
            return ColumnType.TIMESTAMPTZ
        elif self.data_type.lower() == "boolean":
            return ColumnType.BOOLEAN
        elif self.data_type.lower() == "json":
            return ColumnType.JSON
        elif self.data_type.lower() == "jsonb":
            return ColumnType.JSONB
        else:
            return ColumnType.UNKNOWN
    
    @property
    def is_vector_column(self) -> bool:
        """Check if this is a pgvector column."""
        return self.column_type == ColumnType.VECTOR
    
    @property
    def is_text_column(self) -> bool:
        """Check if this is a text-based column."""
        return self.column_type in (ColumnType.TEXT, ColumnType.VARCHAR)
    
    def __str__(self) -> str:
        result = f"{self.name} {self.data_type}"
        if self.vector_dimension:
            result += f"({self.vector_dimension})"
        elif self.max_length:
            result += f"({self.max_length})"
        if not self.is_nullable:
            result += " NOT NULL"
        if self.default_value:
            result += f" DEFAULT {self.default_value}"
        return result


@dataclass
class IndexInfo:
    """Information about a database index."""
    
    name: str
    table_schema: str
    table_name: str
    columns: List[str]
    is_unique: bool
    is_primary: bool
    index_type: str
    definition: str
    
    @property
    def is_vector_index(self) -> bool:
        """Check if this is a pgvector index."""
        return "vector" in self.index_type.lower() or "ivfflat" in self.index_type.lower()


@dataclass
class TableInfo:
    """Information about a database table."""
    
    schema: str
    name: str
    columns: Dict[str, ColumnInfo]
    indexes: List[IndexInfo]
    primary_key: List[str] = None  # Primary key column names
    row_count: Optional[int] = None
    
    # Table metadata
    table_type: str = "BASE TABLE" 
    owner: Optional[str] = None
    has_oids: bool = False
    
    def __post_init__(self):
        if self.primary_key is None:
            self.primary_key = []
    
    @property
    def full_name(self) -> str:
        """Get the fully qualified table name."""
        return f"{self.schema}.{self.name}"
    
    @property
    def vector_columns(self) -> Dict[str, ColumnInfo]:
        """Get all vector columns in the table."""
        return {name: col for name, col in self.columns.items() if col.is_vector_column}
    
    @property
    def text_columns(self) -> Dict[str, ColumnInfo]:
        """Get all text columns in the table.""" 
        return {name: col for name, col in self.columns.items() if col.is_text_column}
    
    def has_column(self, column_name: str) -> bool:
        """Check if table has a specific column."""
        return column_name in self.columns
    
    def get_column(self, column_name: str) -> Optional[ColumnInfo]:
        """Get column information by name."""
        return self.columns.get(column_name)
    
    def get_vector_dimension(self, column_name: str) -> Optional[int]:
        """Get vector dimension for a specific column."""
        col = self.get_column(column_name)
        return col.vector_dimension if col and col.is_vector_column else None


class SchemaIntrospector:
    """Database schema introspection utilities."""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
    
    async def get_table_info(self, schema: str, table: str) -> Optional[TableInfo]:
        """Get complete information about a table."""
        try:
            # Check if table exists
            exists = await self.table_exists(schema, table)
            if not exists:
                return None
            
            # Get columns
            columns = await self.get_columns(schema, table)
            
            # Get indexes
            indexes = await self.get_indexes(schema, table)
            
            # Get table metadata
            metadata = await self.get_table_metadata(schema, table)
            
            return TableInfo(
                schema=schema,
                name=table,
                columns=columns,
                indexes=indexes,
                **metadata
            )
            
        except Exception as e:
            logger.error(f"Error getting table info for {schema}.{table}: {e}")
            raise SchemaError(f"Failed to get table info: {e}") from e
    
    async def table_exists(self, schema: str, table: str) -> bool:
        """Check if a table exists."""
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = $1 AND table_name = $2
            )
        """
        
        try:
            result = await self.pool.fetchval(query, schema, table)
            return bool(result)
        except Exception as e:
            logger.error(f"Error checking table existence for {schema}.{table}: {e}")
            raise DatabaseError(f"Failed to check table existence: {e}") from e
    
    async def get_columns(self, schema: str, table: str) -> Dict[str, ColumnInfo]:
        """Get all columns for a table."""
        query = """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable::boolean,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.ordinal_position,
                c.udt_name,
                c.is_generated::boolean,
                c.generation_expression
            FROM information_schema.columns c
            WHERE c.table_schema = $1 AND c.table_name = $2
            ORDER BY c.ordinal_position
        """
        
        try:
            rows = await self.pool.fetch(query, schema, table)
            columns = {}
            
            for row in rows:
                # Get vector dimension if this is a vector column
                vector_dim = None
                if row["udt_name"] == "vector":
                    vector_dim = await self.get_vector_dimension(schema, table, row["column_name"])
                
                col_info = ColumnInfo(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    is_nullable=row["is_nullable"] == "YES",
                    default_value=row["column_default"],
                    max_length=row["character_maximum_length"],
                    numeric_precision=row["numeric_precision"],
                    numeric_scale=row["numeric_scale"],
                    ordinal_position=row["ordinal_position"],
                    udt_name=row["udt_name"],
                    is_generated=row["is_generated"] == "YES",
                    generation_expression=row["generation_expression"],
                    vector_dimension=vector_dim,
                )
                
                columns[col_info.name] = col_info
            
            return columns
            
        except Exception as e:
            logger.error(f"Error getting columns for {schema}.{table}: {e}")
            raise SchemaError(f"Failed to get columns: {e}") from e
    
    async def get_vector_dimension(self, schema: str, table: str, column: str) -> Optional[int]:
        """Get the dimension of a vector column."""
        query = """
            SELECT atttypmod 
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = $1 
            AND c.relname = $2 
            AND a.attname = $3
            AND NOT a.attisdropped
        """
        
        try:
            result = await self.pool.fetchval(query, schema, table, column)
            # atttypmod for vector type stores dimension + 4 (VARHDRSZ)
            return result - 4 if result and result > 4 else None
        except Exception as e:
            logger.warning(f"Could not get vector dimension for {schema}.{table}.{column}: {e}")
            return None
    
    async def get_indexes(self, schema: str, table: str) -> List[IndexInfo]:
        """Get all indexes for a table."""
        query = """
            SELECT 
                i.indexname as index_name,
                i.schemaname as table_schema,
                i.tablename as table_name,
                i.indexdef as definition,
                ix.indisunique as is_unique,
                ix.indisprimary as is_primary,
                am.amname as index_type
            FROM pg_indexes i
            JOIN pg_class c ON c.relname = i.indexname
            JOIN pg_index ix ON ix.indexrelid = c.oid
            JOIN pg_am am ON am.oid = c.relam
            WHERE i.schemaname = $1 AND i.tablename = $2
            ORDER BY i.indexname
        """
        
        try:
            rows = await self.pool.fetch(query, schema, table)
            indexes = []
            
            for row in rows:
                # Extract column names from index definition
                columns = await self.get_index_columns(schema, row["index_name"])
                
                index_info = IndexInfo(
                    name=row["index_name"],
                    table_schema=row["table_schema"],
                    table_name=row["table_name"],
                    columns=columns,
                    is_unique=row["is_unique"],
                    is_primary=row["is_primary"],
                    index_type=row["index_type"],
                    definition=row["definition"],
                )
                
                indexes.append(index_info)
            
            return indexes
            
        except Exception as e:
            logger.error(f"Error getting indexes for {schema}.{table}: {e}")
            raise SchemaError(f"Failed to get indexes: {e}") from e
    
    async def get_index_columns(self, schema: str, index_name: str) -> List[str]:
        """Get column names for an index."""
        query = """
            SELECT a.attname
            FROM pg_index ix
            JOIN pg_class ic ON ic.oid = ix.indexrelid
            JOIN pg_namespace n ON n.oid = ic.relnamespace
            JOIN pg_attribute a ON a.attrelid = ix.indrelid AND a.attnum = ANY(ix.indkey)
            WHERE n.nspname = $1 AND ic.relname = $2
            ORDER BY array_position(ix.indkey, a.attnum)
        """
        
        try:
            rows = await self.pool.fetch(query, schema, index_name)
            return [row["attname"] for row in rows]
        except Exception as e:
            logger.warning(f"Could not get columns for index {schema}.{index_name}: {e}")
            return []
    
    async def get_table_metadata(self, schema: str, table: str) -> Dict[str, Any]:
        """Get additional table metadata."""
        query = """
            SELECT 
                t.table_type,
                c.relowner::regrole::text as owner,
                c.relhasoids as has_oids,
                s.n_tup_ins + s.n_tup_upd + s.n_tup_del as row_count_estimate
            FROM information_schema.tables t
            JOIN pg_class c ON c.relname = t.table_name
            JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = t.table_schema
            LEFT JOIN pg_stat_user_tables s ON s.schemaname = t.table_schema AND s.relname = t.table_name
            WHERE t.table_schema = $1 AND t.table_name = $2
        """
        
        try:
            row = await self.pool.fetchrow(query, schema, table)
            if row:
                return {
                    "table_type": row["table_type"],
                    "owner": row["owner"],
                    "has_oids": row["has_oids"],
                    "row_count": row["row_count_estimate"],
                }
            return {}
        except Exception as e:
            logger.warning(f"Could not get metadata for {schema}.{table}: {e}")
            return {}
    
    async def list_tables(self, schema: Optional[str] = None) -> List[Tuple[str, str]]:
        """List all tables, optionally filtered by schema."""
        if schema:
            query = """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_schema = $1
                AND table_type = 'BASE TABLE'
                ORDER BY table_schema, table_name
            """
            rows = await self.pool.fetch(query, schema)
        else:
            query = """
                SELECT table_schema, table_name
                FROM information_schema.tables
                WHERE table_type = 'BASE TABLE'
                AND table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
                ORDER BY table_schema, table_name
            """
            rows = await self.pool.fetch(query)
        
        return [(row["table_schema"], row["table_name"]) for row in rows]
    
    async def list_schemas(self) -> List[str]:
        """List all available schemas."""
        query = """
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY schema_name
        """
        
        rows = await self.pool.fetch(query)
        return [row["schema_name"] for row in rows]
    
    async def check_pgvector_extension(self) -> Dict[str, Any]:
        """Check pgvector extension status."""
        query = """
            SELECT 
                installed_version,
                default_version,
                comment
            FROM pg_available_extensions
            WHERE name = 'vector'
        """
        
        try:
            row = await self.pool.fetchrow(query)
            if row:
                return {
                    "available": True,
                    "installed": row["installed_version"] is not None,
                    "installed_version": row["installed_version"],
                    "default_version": row["default_version"],
                    "description": row["comment"],
                }
            else:
                return {
                    "available": False,
                    "installed": False,
                }
        except Exception as e:
            logger.error(f"Error checking pgvector extension: {e}")
            return {
                "available": False,
                "installed": False,
                "error": str(e),
            }
    
    async def find_vector_tables(self) -> List[Tuple[str, str, List[str]]]:
        """Find all tables with vector columns."""
        query = """
            SELECT 
                n.nspname as table_schema,
                c.relname as table_name,
                a.attname as column_name
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            JOIN pg_type t ON t.oid = a.atttypid
            WHERE t.typname = 'vector'
            AND NOT a.attisdropped
            AND c.relkind = 'r'
            AND n.nspname NOT IN ('information_schema', 'pg_catalog', 'pg_toast')
            ORDER BY n.nspname, c.relname, a.attname
        """
        
        try:
            rows = await self.pool.fetch(query)
            
            # Group by table
            tables = {}
            for row in rows:
                key = (row["table_schema"], row["table_name"])
                if key not in tables:
                    tables[key] = []
                tables[key].append(row["column_name"])
            
            return [(schema, table, columns) for (schema, table), columns in tables.items()]
            
        except Exception as e:
            logger.error(f"Error finding vector tables: {e}")
            raise SchemaError(f"Failed to find vector tables: {e}") from e 