"""
Tests for v10r.schema.reconciler module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
from datetime import datetime

from v10r.schema.reconciler import (
    SchemaReconciler,
    ReconciliationStatus,
    ReconciliationResult,
)
from v10r.schema.operations import SchemaChange, ChangeType, OperationMode
from v10r.schema.collision_detector import CollisionInfo, CollisionSeverity
from v10r.database.introspection import TableInfo, ColumnInfo
from v10r.database.connection import ConnectionPool
from v10r.exceptions import SchemaError, ColumnCollisionError


class TestReconciliationStatus:
    """Test ReconciliationStatus enum."""
    
    def test_reconciliation_status_values(self):
        """Test ReconciliationStatus enum values."""
        assert ReconciliationStatus.SUCCESS == "success"
        assert ReconciliationStatus.PARTIAL == "partial"
        assert ReconciliationStatus.FAILED == "failed"
        assert ReconciliationStatus.SKIPPED == "skipped"


class TestReconciliationResult:
    """Test ReconciliationResult dataclass."""
    
    def test_reconciliation_result_creation(self):
        """Test ReconciliationResult creation with all fields."""
        changes = [
            SchemaChange(
                change_type=ChangeType.ADD_COLUMN,
                schema="public",
                table="test",
                description="Add test vector column",
                sql="ALTER TABLE...",
                executed=True,
                error=None
            )
        ]
        
        collisions = [
            CollisionInfo(
                schema_name="public",
                table_name="test_table",
                desired_column="test_vector",
                existing_column_type="text",
                collision_type="unrelated_column",
                severity=CollisionSeverity.WARNING,
                details="Column exists but unrelated"
            )
        ]
        
        result = ReconciliationResult(
            status=ReconciliationStatus.SUCCESS,
            database="test_db",
            schema="public",
            table="test_table",
            changes_applied=changes,
            collisions_detected=collisions,
            errors=[],
            execution_time_ms=150.5
        )
        
        assert result.status == ReconciliationStatus.SUCCESS
        assert result.database == "test_db"
        assert result.schema == "public"
        assert result.table == "test_table"
        assert len(result.changes_applied) == 1
        assert len(result.collisions_detected) == 1
        assert result.errors == []
        assert result.execution_time_ms == 150.5
    
    def test_has_critical_collisions_true(self):
        """Test has_critical_collisions property when critical collisions exist."""
        critical_collision = MagicMock()
        critical_collision.is_critical = True
        
        result = ReconciliationResult(
            status=ReconciliationStatus.FAILED,
            database="test",
            schema="public",
            table="test",
            changes_applied=[],
            collisions_detected=[critical_collision],
            errors=[],
            execution_time_ms=0.0
        )
        
        assert result.has_critical_collisions is True
    
    def test_has_critical_collisions_false(self):
        """Test has_critical_collisions property when no critical collisions."""
        warning_collision = MagicMock()
        warning_collision.is_critical = False
        
        result = ReconciliationResult(
            status=ReconciliationStatus.SUCCESS,
            database="test",
            schema="public", 
            table="test",
            changes_applied=[],
            collisions_detected=[warning_collision],
            errors=[],
            execution_time_ms=0.0
        )
        
        assert result.has_critical_collisions is False
    
    def test_successful_changes_count(self):
        """Test successful_changes property."""
        changes = [
            MagicMock(executed=True),
            MagicMock(executed=False),
            MagicMock(executed=True),
        ]
        
        result = ReconciliationResult(
            status=ReconciliationStatus.PARTIAL,
            database="test",
            schema="public",
            table="test",
            changes_applied=changes,
            collisions_detected=[],
            errors=[],
            execution_time_ms=0.0
        )
        
        assert result.successful_changes == 2
    
    def test_failed_changes_count(self):
        """Test failed_changes property."""
        changes = [
            MagicMock(error=None),
            MagicMock(error="Failed to add column"),
            MagicMock(error="Permission denied"),
        ]
        
        result = ReconciliationResult(
            status=ReconciliationStatus.PARTIAL,
            database="test",
            schema="public",
            table="test",
            changes_applied=changes,
            collisions_detected=[],
            errors=[],
            execution_time_ms=0.0
        )
        
        assert result.failed_changes == 2


class TestSchemaReconciler:
    """Test SchemaReconciler class."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool
    
    @pytest.fixture
    def reconciler(self, mock_pool):
        """Create SchemaReconciler instance with mocked dependencies."""
        with patch('v10r.schema.reconciler.SchemaIntrospector') as mock_introspector_cls, \
             patch('v10r.schema.reconciler.ColumnCollisionDetector') as mock_collision_cls, \
             patch('v10r.schema.reconciler.SafeSchemaOperations') as mock_operations_cls, \
             patch('v10r.schema.reconciler.MetadataManager') as mock_metadata_cls:
            
            reconciler = SchemaReconciler(
                pool=mock_pool,
                operation_mode=OperationMode.SAFE,
                timeout_seconds=300
            )
            
            # Store mocked instances for easy access
            reconciler._mock_introspector = mock_introspector_cls.return_value
            reconciler._mock_collision_detector = mock_collision_cls.return_value  
            reconciler._mock_schema_operations = mock_operations_cls.return_value
            reconciler._mock_metadata_manager = mock_metadata_cls.return_value
            
            return reconciler
    
    def test_reconciler_initialization(self, mock_pool):
        """Test SchemaReconciler initialization."""
        reconciler = SchemaReconciler(
            pool=mock_pool,
            operation_mode=OperationMode.FORCE,
            timeout_seconds=600
        )
        
        assert reconciler.pool == mock_pool
        assert reconciler.operation_mode == OperationMode.FORCE
        assert reconciler.timeout_seconds == 600
        assert reconciler.reconciliation_interval == 300
        assert reconciler.null_check_interval == 60
        assert isinstance(reconciler._active_reconciliations, dict)
        assert reconciler._reconciliation_lock is not None
    
    def test_extract_desired_columns(self, reconciler):
        """Test _extract_desired_columns method."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector",
            "model_column": "content_model"
        }
        
        result = reconciler._extract_desired_columns(table_config)
        
        expected = {
            "content_vector": "vector",
            "content_model": "model",
            "content_vector_created_at": "timestamp"
        }
        assert result == expected
    
    def test_extract_desired_columns_with_defaults(self, reconciler):
        """Test _extract_desired_columns with default column names."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content"
        }
        
        result = reconciler._extract_desired_columns(table_config)
        
        # Should generate default column names
        expected = {
            "content_vector": "vector",
            "content_vector_model": "model", 
            "content_vector_created_at": "timestamp"
        }
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_get_dimension_for_config(self, reconciler):
        """Test _get_dimension_for_config method."""
        # Mock the embedding factory with proper configs that match the hardcoded dict
        reconciler.embedding_factory = MagicMock()
        reconciler.embedding_factory.configs = {
            "openai-small": MagicMock(dimensions=1536),
            "infinity_static": MagicMock(dimensions=768),
            "default": MagicMock(dimensions=768)
        }
        
        # Test existing hardcoded configs
        result = await reconciler._get_dimension_for_config("openai-small")
        assert result == 1536
        
        result = await reconciler._get_dimension_for_config("infinity_static")
        assert result == 768
        
        # Test default fallback
        result = await reconciler._get_dimension_for_config("unknown_config")
        assert result == 768
    
    @pytest.mark.asyncio
    async def test_resolve_column_names_no_collisions(self, reconciler):
        """Test _resolve_column_names with no collisions."""
        table_info = MagicMock()
        # Use the format that _extract_desired_columns returns: column_name -> column_type
        desired_columns = {"content_vector": "vector", "content_model": "model"}
        collisions = {}
        
        result = await reconciler._resolve_column_names(
            table_info, desired_columns, collisions
        )
        
        # Should return column_type -> column_name format
        expected = {"vector": "content_vector", "model": "content_model"}
        assert result == expected

    @pytest.mark.asyncio
    async def test_resolve_column_names_with_collisions(self, reconciler):
        """Test _resolve_column_names with collisions."""
        table_info = MagicMock()
        # Use the format that _extract_desired_columns returns: column_name -> column_type
        desired_columns = {"content_vector": "vector", "content_model": "model"}
        
        collision = MagicMock()
        collision.desired_column = "content_vector"
        collision.collision_type = MagicMock()
        collision.collision_type.value = "unrelated_column"
        collision.severity = MagicMock()
        
        collisions = {"content_vector": collision}
        
        # Mock the _log_collision_resolution method
        reconciler._log_collision_resolution = AsyncMock()
        
        result = await reconciler._resolve_column_names(
            table_info, desired_columns, collisions
        )
        
        # Should return column_type -> column_name format with resolved names
        assert "vector" in result
        assert "model" in result
        # Vector column should be resolved due to collision
        assert result["model"] == "content_model"  # No collision for model
        
        # Should have logged the collision resolution
        reconciler._log_collision_resolution.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_table_config_success(self, reconciler):
        """Test reconcile_table_config with successful reconciliation."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector",
            "embedding_config": "openai"
        }
        
        # Mock table exists
        table_info = MagicMock()
        reconciler.introspector.get_table_info = AsyncMock(return_value=table_info)
        
        # Mock collision detection
        reconciler.collision_detector.check_multiple_collisions = AsyncMock(return_value={})
        
        # Mock schema changes calculation
        reconciler._calculate_schema_changes = AsyncMock(return_value=[])
        
        # Mock metadata update
        reconciler._update_metadata_registry = AsyncMock()
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.SUCCESS
        assert result.database == "test_db"
        assert result.schema == "public"
        assert result.table == "test_table"
        assert len(result.errors) == 0

    @pytest.mark.asyncio
    async def test_reconcile_table_config_table_not_exists(self, reconciler):
        """Test reconcile_table_config when table doesn't exist."""
        table_config = {
            "schema": "public",
            "table": "missing_table",
            "text_column": "content"
        }
        
        # Mock table doesn't exist
        reconciler.introspector.get_table_info = AsyncMock(return_value=None)
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.FAILED
        assert "does not exist" in result.errors[0]

    @pytest.mark.asyncio
    async def test_reconcile_table_config_critical_collision(self, reconciler):
        """Test reconcile_table_config with critical collision in SAFE mode."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector"
        }
        
        # Mock table exists
        table_info = MagicMock()
        reconciler.introspector.get_table_info = AsyncMock(return_value=table_info)
        
        # Mock critical collision
        critical_collision = MagicMock()
        critical_collision.is_critical = True
        critical_collision.desired_column = "content_vector"
        
        reconciler.collision_detector.check_multiple_collisions = AsyncMock(
            return_value={"content_vector": critical_collision}
        )
        
        # Mock logging
        reconciler._log_critical_collisions = AsyncMock()
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.FAILED
        assert "Critical collisions detected" in result.errors[0]
        reconciler._log_critical_collisions.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconcile_table_config_concurrent_protection(self, reconciler):
        """Test reconcile_table_config prevents concurrent reconciliation."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content"
        }
        
        # Set up concurrent reconciliation
        reconciler._active_reconciliations["test_db.public.test_table"] = True
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.SKIPPED
        assert "already in progress" in result.errors[0]

    @pytest.mark.asyncio
    async def test_reconcile_all_tables(self, reconciler):
        """Test reconcile_all_tables method."""
        config = {
            "databases": [
                {
                    "name": "test_db",
                    "tables": [
                        {
                            "schema": "public",
                            "table": "table1",
                            "text_column": "content"
                        },
                        {
                            "schema": "public", 
                            "table": "table2",
                            "text_column": "description"
                        }
                    ]
                }
            ]
        }
        
        # Mock individual reconciliation
        mock_result1 = MagicMock()
        mock_result1.status = MagicMock()
        mock_result2 = MagicMock() 
        mock_result2.status = MagicMock()
        
        reconciler.reconcile_table_config = AsyncMock(side_effect=[mock_result1, mock_result2])
        
        results = await reconciler.reconcile_all_tables(config)
        
        assert len(results) == 2
        assert reconciler.reconcile_table_config.call_count == 2

    @pytest.mark.asyncio
    async def test_check_null_vectors(self, reconciler):
        """Test check_null_vectors method."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector"
        }
        
        # Mock column mapping
        reconciler._get_column_mapping = AsyncMock(return_value={
            "vector": "content_vector"
        })
        
        # Mock null count and total count
        reconciler._count_null_vectors = AsyncMock(return_value=50)
        reconciler._count_total_rows = AsyncMock(return_value=100)
        
        # Mock update methods
        reconciler._update_null_vector_status = AsyncMock()
        reconciler._trigger_bulk_vectorization = AsyncMock()
        
        result = await reconciler.check_null_vectors("test_db", table_config)
        
        assert result["null_vectors"] == 50
        assert result["total_rows"] == 100

    @pytest.mark.asyncio
    async def test_count_null_vectors(self, reconciler):
        """Test _count_null_vectors method."""
        # Mock database connection and cursor
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=25)  # Return actual value, not mock
        
        # Use helper method to properly mock the pool
        self.setup_pool_mock(reconciler, mock_conn)
        
        result = await reconciler._count_null_vectors("test_db", "public", "test_table", "content_vector")
        
        assert result == 25

    @pytest.mark.asyncio
    async def test_count_total_rows(self, reconciler):
        """Test _count_total_rows method."""
        # Mock database connection and cursor
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=100)  # Return actual value, not mock
        
        # Use helper method to properly mock the pool
        self.setup_pool_mock(reconciler, mock_conn)
        
        result = await reconciler._count_total_rows("test_db", "public", "test_table")
        
        assert result == 100

    @pytest.mark.asyncio
    async def test_calculate_schema_changes(self, reconciler):
        """Test _calculate_schema_changes method."""
        table_info = MagicMock()
        table_config = {
            "text_column": "content",
            "vector_column": "content_vector",
            "embedding_config": "openai"  # Embedding config goes in table_config
        }
        resolved_columns = {"text": "content", "vector": "content_vector"}
        
        # Mock operations to return a change
        reconciler.schema_operations.create_vector_column = AsyncMock(return_value=MagicMock())
        
        # Mock table info to show missing vector column
        table_info.columns = {"content": MagicMock()}  # Missing vector column
        table_info.has_column = MagicMock(return_value=False)  # Vector column doesn't exist
        
        changes = await reconciler._calculate_schema_changes(
            table_info, table_config, resolved_columns
        )
        
        # Should return at least one change for missing vector column
        assert len(changes) >= 1

    @pytest.mark.asyncio
    async def test_get_column_mapping(self, reconciler):
        """Test _get_column_mapping method."""
        table_config = {
            "text_column": "content",
            "vector_column": "content_vector"
        }
        
        result = await reconciler._get_column_mapping(table_config)
        
        assert "text" in result
        assert "vector" in result
        assert result["text"] == "content"
        assert result["vector"] == "content_vector"

    @pytest.mark.asyncio
    async def test_update_metadata_registry(self, reconciler):
        """Test _update_metadata_registry method."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content"
        }
        resolved_columns = {"text": "content", "vector": "content_vector"}
        
        # Mock metadata manager
        reconciler.metadata_manager = MagicMock()
        reconciler.metadata_manager.update_table_metadata = AsyncMock()
        
        await reconciler._update_metadata_registry("test_db", table_config, resolved_columns)
        
        reconciler.metadata_manager.update_table_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_log_collision_resolution(self, reconciler):
        """Test _log_collision_resolution method."""
        collision = CollisionInfo(
            schema_name="public",
            table_name="test_table",
            desired_column="content_vector",
            existing_column_type="text",
            collision_type="unrelated_column",
            severity=CollisionSeverity.WARNING,
            details="Column exists but unrelated"
        )
        
        # This should not raise an exception
        await reconciler._log_collision_resolution(collision, "content_vector_v10r")

    @pytest.mark.asyncio
    async def test_log_critical_collisions(self, reconciler):
        """Test _log_critical_collisions method."""
        collision = MagicMock()
        collision.desired_column = "content_vector"
        collision.collision_type = MagicMock()
        collision.collision_type.value = "type_mismatch"  # Add value attribute
        
        collisions = {"content_vector": collision}
        
        # This should not raise an exception
        await reconciler._log_critical_collisions(collisions)

    @pytest.mark.asyncio
    async def test_update_null_vector_status(self, reconciler):
        """Test _update_null_vector_status method."""
        # Mock metadata manager
        reconciler.metadata_manager = MagicMock()
        reconciler.metadata_manager.update_null_vector_status = AsyncMock()
        
        await reconciler._update_null_vector_status(
            "test_db", "public", "test_table", 50, 100
        )
        
        reconciler.metadata_manager.update_null_vector_status.assert_called_once_with(
            "test_db", "public", "test_table", 50, 100
        )

    @pytest.mark.asyncio
    async def test_trigger_bulk_vectorization(self, reconciler):
        """Test _trigger_bulk_vectorization method."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        
        # Use helper method to properly mock the pool
        self.setup_pool_mock(reconciler, mock_conn)
        
        await reconciler._trigger_bulk_vectorization(
            "test_db", "public", "test_table", "content_vector"
        )
        
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_reconciliation_summary(self, reconciler):
        """Test _get_reconciliation_summary method."""
        results = [
            MagicMock(status=ReconciliationStatus.SUCCESS),
            MagicMock(status=ReconciliationStatus.PARTIAL),
            MagicMock(status=ReconciliationStatus.FAILED)
        ]
        
        summary = await reconciler._get_reconciliation_summary(results)
        
        assert "successful_tables" in summary
        assert summary["successful_tables"] == 1

    @pytest.mark.asyncio
    async def test_reconcile_table_config_with_changes(self, reconciler):
        """Test reconcile_table_config when changes need to be applied."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector",
            "embedding_config": "openai"
        }
        
        # Mock table exists
        table_info = MagicMock()
        reconciler.introspector.get_table_info = AsyncMock(return_value=table_info)
        
        # Mock collision detection
        reconciler.collision_detector.check_multiple_collisions = AsyncMock(return_value={})
        
        # Mock schema changes needed
        mock_change = MagicMock(executed=True, error=None)
        reconciler._calculate_schema_changes = AsyncMock(return_value=[mock_change])
        
        # Mock schema operations
        reconciler.schema_operations.execute_batch = AsyncMock(return_value=[mock_change])
        
        # Mock metadata update
        reconciler._update_metadata_registry = AsyncMock()
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.SUCCESS
        assert result.successful_changes == 1

    @pytest.mark.asyncio
    async def test_reconcile_table_config_partial_failure(self, reconciler):
        """Test reconcile_table_config with partial failure."""
        table_config = {
            "schema": "public",
            "table": "test_table",
            "text_column": "content",
            "vector_column": "content_vector"
        }
        
        # Mock table exists
        table_info = MagicMock()
        reconciler.introspector.get_table_info = AsyncMock(return_value=table_info)
        
        # Mock collision detection
        reconciler.collision_detector.check_multiple_collisions = AsyncMock(return_value={})
        
        # Mock mixed success/failure changes
        success_change = MagicMock(executed=True, error=None)
        failed_change = MagicMock(executed=False, error="Permission denied")
        
        reconciler._calculate_schema_changes = AsyncMock(return_value=[success_change, failed_change])
        reconciler.schema_operations.execute_batch = AsyncMock(return_value=[success_change, failed_change])
        
        # Mock metadata update
        reconciler._update_metadata_registry = AsyncMock()
        
        result = await reconciler.reconcile_table_config("test_db", table_config)
        
        assert result.status == ReconciliationStatus.PARTIAL
        assert result.successful_changes == 1
        assert result.failed_changes == 1

    def setup_pool_mock(self, reconciler, mock_conn):
        """Helper method to properly mock the pool.acquire() async context manager."""
        # Create a real async context manager class
        class MockAsyncContextManager:
            def __init__(self, conn):
                self.conn = conn
                
            async def __aenter__(self):
                return self.conn
                
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None
        
        # Make pool.acquire() return the async context manager instance
        reconciler.pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        return MockAsyncContextManager(mock_conn)
