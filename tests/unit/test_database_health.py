"""
Tests for v10r.database.health module.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from v10r.database.health import (
    DatabaseHealthChecker,
    HealthCheckResult,
    HealthStatus,
)
from v10r.database.connection import ConnectionPool
from v10r.exceptions import DatabaseError


class TestHealthStatus:
    """Test HealthStatus enum."""
    
    def test_health_status_values(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.WARNING == "warning"
        assert HealthStatus.CRITICAL == "critical"
        assert HealthStatus.UNKNOWN == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""
    
    def test_health_check_result_creation(self):
        """Test creating a HealthCheckResult."""
        result = HealthCheckResult(
            name="test_check",
            status=HealthStatus.HEALTHY,
            message="Test passed",
            details={"key": "value"},
            duration_ms=123.45,
            timestamp=1234567890.0,
        )
        
        assert result.name == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Test passed"
        assert result.details == {"key": "value"}
        assert result.duration_ms == 123.45
        assert result.timestamp == 1234567890.0
    
    def test_is_healthy_property(self):
        """Test is_healthy property."""
        healthy_result = HealthCheckResult(
            name="test", status=HealthStatus.HEALTHY, message="", 
            details={}, duration_ms=0, timestamp=0
        )
        warning_result = HealthCheckResult(
            name="test", status=HealthStatus.WARNING, message="", 
            details={}, duration_ms=0, timestamp=0
        )
        
        assert healthy_result.is_healthy is True
        assert warning_result.is_healthy is False
    
    def test_is_critical_property(self):
        """Test is_critical property."""
        critical_result = HealthCheckResult(
            name="test", status=HealthStatus.CRITICAL, message="", 
            details={}, duration_ms=0, timestamp=0
        )
        healthy_result = HealthCheckResult(
            name="test", status=HealthStatus.HEALTHY, message="", 
            details={}, duration_ms=0, timestamp=0
        )
        
        assert critical_result.is_critical is True
        assert healthy_result.is_critical is False


class TestDatabaseHealthChecker:
    """Test DatabaseHealthChecker class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create a mock connection pool."""
        pool = MagicMock(spec=ConnectionPool)
        pool.is_initialized = True
        pool.is_closed = False
        pool.config = MagicMock()
        pool.config.min_size = 1
        pool.config.max_size = 10
        return pool
    
    @pytest.fixture
    def health_checker(self, mock_pool):
        """Create a DatabaseHealthChecker instance."""
        with patch('v10r.database.health.SchemaIntrospector'):
            return DatabaseHealthChecker(mock_pool, "test_db")
    
    def test_init(self, mock_pool):
        """Test DatabaseHealthChecker initialization."""
        with patch('v10r.database.health.SchemaIntrospector') as mock_introspector:
            checker = DatabaseHealthChecker(mock_pool, "test_db")
            assert checker.pool is mock_pool
            assert checker.name == "test_db"
            mock_introspector.assert_called_once_with(mock_pool)
    
    def test_init_default_name(self, mock_pool):
        """Test DatabaseHealthChecker initialization with default name."""
        with patch('v10r.database.health.SchemaIntrospector'):
            checker = DatabaseHealthChecker(mock_pool)
            assert checker.name == "default"
    
    @pytest.mark.asyncio
    async def test_check_connectivity_success(self, health_checker, mock_pool):
        """Test successful connectivity check."""
        # Mock connection and result
        mock_conn = AsyncMock()
        mock_conn.fetchrow.return_value = {
            "version": "PostgreSQL 16.0",
            "current_database": "test_db",
            "current_user": "test_user"
        }
        
        mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        result = await health_checker.check_connectivity()
        
        assert result.name == "connectivity"
        assert result.status == HealthStatus.HEALTHY
        assert "successful" in result.message
        assert result.details["database"] == "test_db"
        assert result.details["user"] == "test_user"
        assert result.details["version"] == "PostgreSQL 16.0"
        assert result.duration_ms > 0
        assert result.timestamp > 0
    
    @pytest.mark.asyncio
    async def test_check_connectivity_failure(self, health_checker, mock_pool):
        """Test connectivity check failure."""
        # Mock connection failure
        mock_pool.acquire.side_effect = Exception("Connection failed")
        
        result = await health_checker.check_connectivity()
        
        assert result.name == "connectivity"
        assert result.status == HealthStatus.CRITICAL
        assert "Connection failed" in result.message
        assert "error" in result.details
        assert result.duration_ms >= 0
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_healthy(self, health_checker):
        """Test pgvector extension check when healthy."""
        extension_info = {
            "available": True,
            "installed": True,
            "version": "0.5.0"
        }
        health_checker.introspector.check_pgvector_extension = AsyncMock(return_value=extension_info)
        
        result = await health_checker.check_pgvector_extension()
        
        assert result.name == "pgvector_extension"
        assert result.status == HealthStatus.HEALTHY
        assert "ready" in result.message
        assert result.details == extension_info
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_not_available(self, health_checker):
        """Test pgvector extension check when not available."""
        extension_info = {
            "available": False,
            "installed": False,
            "version": None
        }
        health_checker.introspector.check_pgvector_extension = AsyncMock(return_value=extension_info)
        
        result = await health_checker.check_pgvector_extension()
        
        assert result.name == "pgvector_extension"
        assert result.status == HealthStatus.CRITICAL
        assert "not available" in result.message
        assert result.details == extension_info
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_not_installed(self, health_checker):
        """Test pgvector extension check when available but not installed."""
        extension_info = {
            "available": True,
            "installed": False,
            "version": None
        }
        health_checker.introspector.check_pgvector_extension = AsyncMock(return_value=extension_info)
        
        result = await health_checker.check_pgvector_extension()
        
        assert result.name == "pgvector_extension"
        assert result.status == HealthStatus.WARNING
        assert "not installed" in result.message
        assert result.details == extension_info
    
    @pytest.mark.asyncio
    async def test_check_pgvector_extension_error(self, health_checker):
        """Test pgvector extension check with error."""
        health_checker.introspector.check_pgvector_extension = AsyncMock(
            side_effect=Exception("Check failed")
        )
        
        result = await health_checker.check_pgvector_extension()
        
        assert result.name == "pgvector_extension"
        assert result.status == HealthStatus.CRITICAL
        assert "failed" in result.message
        assert "error" in result.details
    
    @pytest.mark.asyncio
    async def test_check_connection_pool_healthy(self, health_checker, mock_pool):
        """Test connection pool check when healthy."""
        mock_pool.is_initialized = True
        mock_pool.is_closed = False
        
        result = await health_checker.check_connection_pool()
        
        assert result.name == "connection_pool"
        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message
        assert result.details["is_initialized"] is True
        assert result.details["is_closed"] is False
        assert result.details["min_size"] == 1
        assert result.details["max_size"] == 10
    
    @pytest.mark.asyncio
    async def test_check_connection_pool_closed(self, health_checker, mock_pool):
        """Test connection pool check when closed."""
        mock_pool.is_initialized = True
        mock_pool.is_closed = True
        
        result = await health_checker.check_connection_pool()
        
        assert result.name == "connection_pool"
        assert result.status == HealthStatus.CRITICAL
        assert "closed" in result.message
    
    @pytest.mark.asyncio
    async def test_check_connection_pool_not_initialized(self, health_checker, mock_pool):
        """Test connection pool check when not initialized."""
        mock_pool.is_initialized = False
        mock_pool.is_closed = False
        mock_pool.initialize = AsyncMock()
        
        result = await health_checker.check_connection_pool()
        
        assert result.name == "connection_pool"
        assert result.status == HealthStatus.WARNING
        assert "not initialized" in result.message
        mock_pool.initialize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_connection_pool_error(self, health_checker, mock_pool):
        """Test connection pool check with error."""
        mock_pool.is_initialized = False
        mock_pool.initialize = AsyncMock(side_effect=Exception("Initialize failed"))
        
        result = await health_checker.check_connection_pool()
        
        assert result.name == "connection_pool"
        assert result.status == HealthStatus.CRITICAL
        assert "failed" in result.message
        assert "error" in result.details
    
    @pytest.mark.asyncio
    async def test_check_basic_query_performance_healthy(self, health_checker, mock_pool):
        """Test basic query performance check when healthy."""
        mock_pool.fetchval = AsyncMock(return_value=42)
        
        with patch('time.time', side_effect=[1000.0, 1000.05, 1000.1, 1000.1, 1000.1]):  # start, query_start, query_end, duration_end, timestamp
            result = await health_checker.check_basic_query_performance()
        
        assert result.name == "query_performance"
        assert result.status == HealthStatus.HEALTHY
        assert "good" in result.message
        assert abs(result.details["query_duration_ms"] - 50.0) < 0.1
    
    @pytest.mark.asyncio
    async def test_check_basic_query_performance_slow(self, health_checker, mock_pool):
        """Test basic query performance check when slow."""
        mock_pool.fetchval = AsyncMock(return_value=42)
        
        with patch('time.time', side_effect=[1000.0, 1000.0, 1000.5, 1000.5, 1000.5, 1000.5]):  # start, query_start, query_end, duration_end, timestamp
            result = await health_checker.check_basic_query_performance()
        
        assert result.name == "query_performance"
        assert result.status == HealthStatus.WARNING
        assert "slow" in result.message
        assert result.details["query_duration_ms"] == 500.0
    
    @pytest.mark.asyncio
    async def test_check_basic_query_performance_very_slow(self, health_checker, mock_pool):
        """Test basic query performance check when very slow."""
        mock_pool.fetchval = AsyncMock(return_value=42)
        
        with patch('time.time', side_effect=[1000.0, 1000.0, 1002.0, 1002.0, 1002.0, 1002.0]):  # start, query_start, query_end, duration_end, timestamp
            result = await health_checker.check_basic_query_performance()
        
        assert result.name == "query_performance"
        assert result.status == HealthStatus.CRITICAL
        assert "very slow" in result.message
        assert result.details["query_duration_ms"] == 2000.0
    
    @pytest.mark.asyncio
    async def test_check_basic_query_performance_error(self, health_checker, mock_pool):
        """Test basic query performance check with error."""
        mock_pool.fetchval = AsyncMock(side_effect=Exception("Query failed"))
        
        with patch('time.time', side_effect=[1000.0, 1000.0, 1000.1, 1000.1, 1000.1]):  # start, query_start, error_duration, timestamp, logger_time
            result = await health_checker.check_basic_query_performance()
        
        assert result.name == "query_performance"
        assert result.status == HealthStatus.CRITICAL
        assert "failed" in result.message
        assert "error" in result.details
    
    @pytest.mark.asyncio
    async def test_check_all_success(self, health_checker):
        """Test check_all when all checks succeed."""
        # Mock all individual check methods
        health_checker.check_connectivity = AsyncMock(return_value=HealthCheckResult(
            name="connectivity", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=10, timestamp=time.time()
        ))
        health_checker.check_pgvector_extension = AsyncMock(return_value=HealthCheckResult(
            name="pgvector_extension", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=20, timestamp=time.time()
        ))
        health_checker.check_connection_pool = AsyncMock(return_value=HealthCheckResult(
            name="connection_pool", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=30, timestamp=time.time()
        ))
        health_checker.check_basic_query_performance = AsyncMock(return_value=HealthCheckResult(
            name="query_performance", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=40, timestamp=time.time()
        ))
        
        results = await health_checker.check_all()
        
        assert len(results) == 4
        assert "connectivity" in results
        assert "pgvector_extension" in results
        assert "connection_pool" in results
        assert "query_performance" in results
        assert all(result.status == HealthStatus.HEALTHY for result in results.values())
    
    @pytest.mark.asyncio
    async def test_check_all_with_exception(self, health_checker):
        """Test check_all when some checks raise exceptions."""
        # Mock some methods to succeed and others to fail
        health_checker.check_connectivity = AsyncMock(return_value=HealthCheckResult(
            name="connectivity", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=10, timestamp=time.time()
        ))
        health_checker.check_pgvector_extension = AsyncMock(
            side_effect=Exception("pgvector check failed")
        )
        health_checker.check_connection_pool = AsyncMock(return_value=HealthCheckResult(
            name="connection_pool", status=HealthStatus.HEALTHY, message="OK",
            details={}, duration_ms=30, timestamp=time.time()
        ))
        health_checker.check_basic_query_performance = AsyncMock(
            side_effect=DatabaseError("Query check failed")
        )
        
        results = await health_checker.check_all()
        
        assert len(results) == 4
        assert results["connectivity"].status == HealthStatus.HEALTHY
        assert results["connection_pool"].status == HealthStatus.HEALTHY
        
        # Check that exceptions were converted to critical results
        pgvector_result = None
        query_result = None
        for name, result in results.items():
            if "pgvector check failed" in result.message:
                pgvector_result = result
            elif "Query check failed" in result.message:
                query_result = result
        
        assert pgvector_result is not None
        assert pgvector_result.status == HealthStatus.CRITICAL
        assert query_result is not None
        assert query_result.status == HealthStatus.CRITICAL 