"""
Database connection management for v10r.

Provides async PostgreSQL connection pooling with health checks,
automatic reconnection, and connection lifecycle management.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Optional, Any, AsyncIterator, List
from urllib.parse import urlparse, parse_qs

import asyncpg
from pydantic import BaseModel, Field, field_validator

from ..exceptions import DatabaseConnectionError, DatabaseConfigurationError


logger = logging.getLogger(__name__)


class ConnectionConfig(BaseModel):
    """Database connection configuration."""
    
    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    
    # Connection pool settings
    min_size: int = Field(5, description="Minimum connections in pool")
    max_size: int = Field(20, description="Maximum connections in pool")
    
    # Connection settings
    command_timeout: float = Field(60.0, description="Command timeout in seconds")
    server_settings: Dict[str, str] = Field(
        default_factory=lambda: {"application_name": "v10r"},
        description="PostgreSQL server settings"
    )
    
    # SSL settings
    ssl_mode: Optional[str] = Field(None, description="SSL mode")
    ssl_ca: Optional[str] = Field(None, description="SSL CA certificate path")
    ssl_cert: Optional[str] = Field(None, description="SSL certificate path")
    ssl_key: Optional[str] = Field(None, description="SSL key path")
    
    @field_validator('database')
    @classmethod
    def validate_database(cls, v):
        if not v or not v.strip():
            raise ValueError("Database name is required")
        return v
    
    @property
    def pool_size(self) -> int:
        """Alias for max_size for backward compatibility."""
        return self.max_size
    
    @classmethod
    def from_url(cls, url: str) -> "ConnectionConfig":
        """Create configuration from database URL."""
        parsed = urlparse(url)
        
        if parsed.scheme not in ("postgresql", "postgres"):
            raise DatabaseConfigurationError(f"Invalid database URL scheme: {parsed.scheme}")
        
        if not parsed.path or parsed.path == "/":
            raise DatabaseConfigurationError("Database name is required")
        
        # Parse query parameters for SSL settings
        query_params = parse_qs(parsed.query) if parsed.query else {}
        
        config_data = {
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
            "database": parsed.path.lstrip("/"),
            "user": parsed.username or "",
            "password": parsed.password or "",
        }
        
        # Handle SSL parameters from query string
        if "sslmode" in query_params:
            config_data["ssl_mode"] = query_params["sslmode"][0]
        else:
            config_data["ssl_mode"] = "prefer"  # Default SSL mode
            
        if "sslcert" in query_params:
            config_data["ssl_cert"] = query_params["sslcert"][0]
        if "sslkey" in query_params:
            config_data["ssl_key"] = query_params["sslkey"][0]
        if "sslrootcert" in query_params:
            config_data["ssl_ca"] = query_params["sslrootcert"][0]
        
        return cls(**config_data)
    
    def validate(self) -> None:
        """Validate the configuration."""
        # Pydantic v2 validation happens automatically
        # This method is kept for backward compatibility
        pass
    
    def to_connection_kwargs(self) -> Dict[str, Any]:
        """Convert to asyncpg connection kwargs."""
        kwargs = {
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "user": self.user,
            "password": self.password,
            "command_timeout": self.command_timeout,
            "server_settings": self.server_settings,
        }
        
        # Add SSL settings if configured
        if self.ssl_mode:
            kwargs["ssl"] = self.ssl_mode
        if self.ssl_ca:
            kwargs["ssl_ca"] = self.ssl_ca
        if self.ssl_cert:
            kwargs["ssl_cert"] = self.ssl_cert
        if self.ssl_key:
            kwargs["ssl_key"] = self.ssl_key
        
        return kwargs


class ConnectionPool:
    """Async PostgreSQL connection pool wrapper."""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self._pool: Optional[asyncpg.Pool] = None
        self._lock = asyncio.Lock()
        
    async def connect(self) -> None:
        """Connect to database and initialize pool. Alias for initialize()."""
        await self.initialize()
        
    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self._lock:
            if self._pool is not None:
                return
                
            try:
                logger.info(
                    f"Initializing connection pool to {self.config.host}:{self.config.port}"
                    f"/{self.config.database} (min={self.config.min_size}, max={self.config.max_size})"
                )
                
                self._pool = await asyncpg.create_pool(
                    **self.config.to_connection_kwargs(),
                    min_size=self.config.min_size,
                    max_size=self.config.max_size,
                )
                
                logger.info("Connection pool initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize connection pool: {e}")
                raise DatabaseConnectionError(f"Failed to initialize connection pool: {e}") from e
    
    async def close(self) -> None:
        """Close the connection pool."""
        async with self._lock:
            if self._pool is not None:
                logger.info("Closing connection pool")
                await self._pool.close()
                self._pool = None
    
    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from the pool."""
        if self._pool is None:
            raise DatabaseConnectionError("Pool is not connected")
        
        async with self._pool.acquire() as connection:
            yield connection
    
    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status."""
        async with self.acquire() as conn:
            return await conn.execute(query, *args)
    
    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Fetch all results from a query."""
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Fetch a single row from a query."""
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)
    
    async def fetchval(self, query: str, *args, column: int = 0) -> Any:
        """Fetch a single value from a query."""
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args, column=column)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self._pool is None:
            return {
                "size": 0,
                "free": 0,
                "acquired": 0,
                "initialized": False
            }
        
        return {
            "size": self._pool.get_size(),
            "free": self._pool.get_idle_size(),
            "acquired": self._pool.get_size() - self._pool.get_idle_size(),
            "initialized": True
        }
    
    @property
    def is_initialized(self) -> bool:
        """Check if pool is initialized."""
        return self._pool is not None
    
    @property
    def is_closed(self) -> bool:
        """Check if pool is closed."""
        if self._pool is None:
            return True
        # Handle both real asyncpg.Pool and mock objects
        if hasattr(self._pool, '_closed'):
            return self._pool._closed
        # For mock objects without _closed attribute
        return False


class DatabaseManager:
    """Manages multiple database connections for v10r."""
    
    def __init__(self):
        self._pools: Dict[str, ConnectionPool] = {}
        self._lock = asyncio.Lock()
    
    @property
    def pools(self) -> Dict[str, ConnectionPool]:
        """Access to managed connection pools."""
        return self._pools
    
    def add_database(self, name: str, config: Dict[str, Any]) -> None:
        """Add a database connection (sync version for compatibility)."""
        if name in self._pools:
            raise DatabaseConfigurationError(f"Database '{name}' already exists")
        
        logger.info(f"Adding database connection '{name}'")
        connection_config = ConnectionConfig(**config)
        pool = ConnectionPool(connection_config)
        self._pools[name] = pool
    
    async def add_database_async(self, name: str, config: ConnectionConfig) -> None:
        """Add a database connection and initialize it."""
        async with self._lock:
            if name in self._pools:
                raise DatabaseConfigurationError(f"Database '{name}' already exists")
            
            logger.info(f"Adding database connection '{name}'")
            pool = ConnectionPool(config)
            await pool.initialize()
            self._pools[name] = pool
    
    async def connect_all(self) -> None:
        """Connect all configured databases."""
        async with self._lock:
            for name, pool in self._pools.items():
                if not pool.is_initialized:
                    logger.info(f"Connecting to database '{name}'")
                    await pool.initialize()
    
    async def remove_database(self, name: str) -> None:
        """Remove a database connection."""
        async with self._lock:
            if name not in self._pools:
                raise DatabaseConfigurationError(f"Database '{name}' not found")
            
            logger.info(f"Removing database connection '{name}'")
            await self._pools[name].close()
            del self._pools[name]
    
    def get_pool(self, name: str) -> ConnectionPool:
        """Get a connection pool by name (sync method)."""
        if name not in self._pools:
            raise DatabaseConfigurationError(f"Database '{name}' not found")
        
        return self._pools[name]
    
    async def get_pool_async(self, name: str) -> ConnectionPool:
        """Get a connection pool by name and ensure it's initialized."""
        if name not in self._pools:
            raise DatabaseConfigurationError(f"Database '{name}' not found")
        
        pool = self._pools[name]
        if not pool.is_initialized:
            await pool.initialize()
        
        return pool
    
    @asynccontextmanager
    async def acquire(self, database: str) -> AsyncIterator[asyncpg.Connection]:
        """Acquire a connection from a specific database."""
        pool = await self.get_pool_async(database)
        async with pool.acquire() as conn:
            yield conn
    
    async def close_all(self) -> None:
        """Close all database connections."""
        async with self._lock:
            logger.info("Closing all database connections")
            for name, pool in self._pools.items():
                try:
                    await pool.close()
                except Exception as e:
                    logger.error(f"Error closing pool '{name}': {e}")
            
            self._pools.clear()
    
    def list_databases(self) -> List[str]:
        """List all configured database names."""
        return list(self._pools.keys())
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all managed databases."""
        return {name: pool.get_stats() for name, pool in self._pools.items()}
    
    async def test_connection(self, name: str) -> Dict[str, Any]:
        """Test a database connection and return connection info."""
        try:
            pool = await self.get_pool_async(name)
            async with pool.acquire() as conn:
                # Test basic query
                result = await conn.fetchrow("SELECT version(), current_database(), current_user")
                
                # Check pgvector extension
                pgvector_result = await conn.fetchrow("""
                    SELECT installed_version 
                    FROM pg_available_extensions 
                    WHERE name = 'vector' AND installed_version IS NOT NULL
                """)
                
                return {
                    "status": "connected",
                    "database": result["current_database"],
                    "user": result["current_user"],
                    "version": result["version"],
                    "pgvector_version": pgvector_result["installed_version"] if pgvector_result else None,
                    "pool_size": len(pool._pool._holders) if pool._pool else 0,
                }
                
        except Exception as e:
            logger.error(f"Connection test failed for '{name}': {e}")
            return {
                "status": "failed",
                "error": str(e),
            }
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Perform health checks on all databases."""
        results = {}
        for name in self._pools.keys():
            results[name] = await self.test_connection(name)
        return results
    
    async def disconnect_all(self) -> None:
        """Alias for close_all() for backward compatibility."""
        await self.close_all()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close_all() 