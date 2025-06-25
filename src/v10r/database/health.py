"""
Database health checking for v10r.

Provides health check utilities for monitoring database connectivity,
performance, and v10r-specific requirements.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum

from .connection import ConnectionPool
from .introspection import SchemaIntrospector
from ..exceptions import DatabaseError


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""
    
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: float
    
    @property
    def is_healthy(self) -> bool:
        """Check if the result indicates healthy status."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def is_critical(self) -> bool:
        """Check if the result indicates critical status."""
        return self.status == HealthStatus.CRITICAL


class DatabaseHealthChecker:
    """Database health monitoring for v10r."""
    
    def __init__(self, pool: ConnectionPool, name: str = "default"):
        self.pool = pool
        self.name = name
        self.introspector = SchemaIntrospector(pool)
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        checks = [
            self.check_connectivity(),
            self.check_pgvector_extension(),
            self.check_connection_pool(),
            self.check_basic_query_performance(),
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        health_results = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = f"check_{i}"
                health_results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {result}",
                    details={"error": str(result)},
                    duration_ms=0.0,
                    timestamp=time.time(),
                )
            else:
                health_results[result.name] = result
        
        return health_results
    
    async def check_connectivity(self) -> HealthCheckResult:
        """Check basic database connectivity."""
        start_time = time.time()
        
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("SELECT version(), current_database(), current_user")
                
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name="connectivity",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "database": result["current_database"],
                    "user": result["current_user"],
                    "version": result["version"],
                },
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Database connectivity check failed: {e}")
            
            return HealthCheckResult(
                name="connectivity",
                status=HealthStatus.CRITICAL,
                message=f"Database connection failed: {e}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
    
    async def check_pgvector_extension(self) -> HealthCheckResult:
        """Check pgvector extension status."""
        start_time = time.time()
        
        try:
            extension_info = await self.introspector.check_pgvector_extension()
            duration_ms = (time.time() - start_time) * 1000
            
            if not extension_info["available"]:
                return HealthCheckResult(
                    name="pgvector_extension",
                    status=HealthStatus.CRITICAL,
                    message="pgvector extension is not available",
                    details=extension_info,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
            elif not extension_info["installed"]:
                return HealthCheckResult(
                    name="pgvector_extension",
                    status=HealthStatus.WARNING,
                    message="pgvector extension is available but not installed",
                    details=extension_info,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
            else:
                return HealthCheckResult(
                    name="pgvector_extension",
                    status=HealthStatus.HEALTHY,
                    message="pgvector extension is installed and ready",
                    details=extension_info,
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"pgvector extension check failed: {e}")
            
            return HealthCheckResult(
                name="pgvector_extension",
                status=HealthStatus.CRITICAL,
                message=f"pgvector extension check failed: {e}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
    
    async def check_connection_pool(self) -> HealthCheckResult:
        """Check connection pool health."""
        start_time = time.time()
        
        try:
            if not self.pool.is_initialized:
                await self.pool.initialize()
            
            pool_info = {
                "is_initialized": self.pool.is_initialized,
                "is_closed": self.pool.is_closed,
                "min_size": self.pool.config.min_size,
                "max_size": self.pool.config.max_size,
            }
            
            duration_ms = (time.time() - start_time) * 1000
            
            if self.pool.is_closed:
                status = HealthStatus.CRITICAL
                message = "Connection pool is closed"
            elif not self.pool.is_initialized:
                status = HealthStatus.WARNING
                message = "Connection pool is not initialized"
            else:
                status = HealthStatus.HEALTHY
                message = "Connection pool is healthy"
            
            return HealthCheckResult(
                name="connection_pool",
                status=status,
                message=message,
                details=pool_info,
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Connection pool check failed: {e}")
            
            return HealthCheckResult(
                name="connection_pool",
                status=HealthStatus.CRITICAL,
                message=f"Connection pool check failed: {e}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
    
    async def check_basic_query_performance(self) -> HealthCheckResult:
        """Check basic query performance."""
        start_time = time.time()
        
        try:
            # Simple performance test
            query_start = time.time()
            await self.pool.fetchval("SELECT COUNT(*) FROM pg_stat_activity")
            query_duration = (time.time() - query_start) * 1000
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on query performance
            if query_duration < 100:  # < 100ms
                status = HealthStatus.HEALTHY
                message = "Query performance is good"
            elif query_duration < 1000:  # < 1s
                status = HealthStatus.WARNING
                message = "Query performance is slow"
            else:  # >= 1s
                status = HealthStatus.CRITICAL
                message = "Query performance is very slow"
            
            return HealthCheckResult(
                name="query_performance",
                status=status,
                message=message,
                details={"query_duration_ms": query_duration},
                duration_ms=duration_ms,
                timestamp=time.time(),
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Query performance check failed: {e}")
            
            return HealthCheckResult(
                name="query_performance",
                status=HealthStatus.CRITICAL,
                message=f"Query performance check failed: {e}",
                details={"error": str(e)},
                duration_ms=duration_ms,
                timestamp=time.time(),
            ) 