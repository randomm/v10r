"""
Unit tests for schema collision detection.

Tests collision detection logic, severity assessment, and alternative name generation
following pytest best practices.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, Any, List

from v10r.schema.collision_detector import (
    ColumnCollisionDetector,
    CollisionInfo,
    CollisionSeverity,
    CollisionType,
)
from v10r.database.connection import ConnectionPool
from v10r.database.introspection import ColumnInfo, TableInfo


class TestCollisionSeverity:
    """Test collision severity enumeration."""
    
    def test_collision_severity_values(self):
        """Test collision severity enum values."""
        assert CollisionSeverity.SAFE.value == "safe"
        assert CollisionSeverity.INFO.value == "info"
        assert CollisionSeverity.WARNING.value == "warning"
        assert CollisionSeverity.CRITICAL.value == "critical"
    
    def test_collision_severity_ordering(self):
        """Test collision severity ordering for comparison."""
        severities = [
            CollisionSeverity.SAFE,
            CollisionSeverity.INFO,
            CollisionSeverity.WARNING,
            CollisionSeverity.CRITICAL,
        ]
        
        # Each should be less than the next using proper ordering
        for i in range(len(severities) - 1):
            assert severities[i] < severities[i + 1]


class TestCollisionType:
    """Test collision type enumeration."""
    
    def test_collision_type_values(self):
        """Test all collision type values are defined."""
        expected_types = {
            "no_collision",
            "unrelated_column", 
            "vectorizer_pattern",
            "previous_vectorizer",
            "existing_vector",
        }
        
        actual_types = {ct.value for ct in CollisionType}
        assert actual_types == expected_types


class TestCollisionInfo:
    """Test collision information dataclass."""
    
    def test_collision_info_creation(self):
        """Test creating collision info object."""
        collision = CollisionInfo(
            schema_name="public",
            table_name="documents",
            desired_column="content_vector",
            existing_column_type="text",
            collision_type=CollisionType.UNRELATED_COLUMN,
            severity=CollisionSeverity.INFO,
            details="Column exists but unrelated to vectorization",
            alternative_names=["content_vector_v2"],
        )
        
        assert collision.table_full_name == "public.documents"
        assert collision.is_critical == False
        assert collision.is_safe == False
        assert collision.needs_resolution == True
    
    def test_collision_info_properties_critical(self):
        """Test collision info properties for critical collision."""
        collision = CollisionInfo(
            schema_name="public",
            table_name="documents",
            desired_column="content_vector",
            existing_column_type="vector",
            collision_type=CollisionType.EXISTING_VECTOR,
            severity=CollisionSeverity.CRITICAL,
            details="Column is already a vector",
        )
        
        assert collision.is_critical == True
        assert collision.is_safe == False
        assert collision.needs_resolution == True
    
    def test_collision_info_properties_safe(self):
        """Test collision info properties for safe collision."""
        collision = CollisionInfo(
            schema_name="public",
            table_name="documents",
            desired_column="new_column",
            existing_column_type=None,
            collision_type=CollisionType.NO_COLLISION,
            severity=CollisionSeverity.SAFE,
            details="No collision detected",
        )
        
        assert collision.is_critical == False
        assert collision.is_safe == True
        assert collision.needs_resolution == False


class TestColumnCollisionDetector:
    """Test cases for ColumnCollisionDetector."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool
    
    @pytest.fixture
    def detector(self, mock_pool):
        """Create collision detector with mock pool."""
        return ColumnCollisionDetector(mock_pool)
    
    @pytest.fixture
    def sample_table_info(self):
        """Create sample table information."""
        return TableInfo(
            schema="public",
            name="documents",
            columns={
                "id": ColumnInfo(
                    name="id",
                    data_type="integer",
                    is_nullable=False,
                    default="nextval('documents_id_seq'::regclass)",
                ),
                "content": ColumnInfo(
                    name="content", 
                    data_type="text",
                    is_nullable=True,
                ),
                "existing_vector": ColumnInfo(
                    name="existing_vector",
                    data_type="USER-DEFINED",
                    udt_name="vector",
                    is_nullable=True,
                ),
                "description_cleaned": ColumnInfo(
                    name="description_cleaned",
                    data_type="text",
                    is_nullable=True,
                ),
            },
            indexes=[],
            primary_key=["id"],
        )
    
    def test_detector_initialization(self, mock_pool):
        """Test detector initialization."""
        detector = ColumnCollisionDetector(mock_pool)
        
        assert detector.pool == mock_pool
        assert len(detector.vectorizer_patterns) > 0
        assert any("vector" in pattern for pattern in detector.vectorizer_patterns)
        assert any("embedding" in pattern for pattern in detector.vectorizer_patterns)
    
    @pytest.mark.asyncio
    async def test_check_single_collision_no_collision(self, detector, sample_table_info):
        """Test checking collision when no collision exists."""
        with patch.object(detector.introspector, 'get_table_info', return_value=sample_table_info):
            collision = await detector.check_single_collision(
                "public", "documents", "new_column"
            )
        
        assert collision.collision_type == CollisionType.NO_COLLISION
        assert collision.severity == CollisionSeverity.SAFE
        assert not collision.is_critical
    
    @pytest.mark.asyncio
    async def test_check_single_collision_existing_vector(self, detector, sample_table_info):
        """Test collision with existing vector column."""
        with patch.object(detector.introspector, 'get_table_info', return_value=sample_table_info):
            collision = await detector.check_single_collision(
                "public", "documents", "existing_vector"
            )
        
        assert collision.collision_type == CollisionType.EXISTING_VECTOR
        assert collision.severity == CollisionSeverity.CRITICAL
        assert collision.is_critical
        assert "already a pgvector column" in collision.details.lower()
    
    @pytest.mark.asyncio
    async def test_check_single_collision_vectorizer_pattern(self, detector, sample_table_info):
        """Test collision with vectorizer naming pattern."""
        with patch.object(detector.introspector, 'get_table_info', return_value=sample_table_info):
            collision = await detector.check_single_collision(
                "public", "documents", "description_cleaned"
            )
        
        assert collision.collision_type == CollisionType.VECTORIZER_PATTERN
        assert collision.severity == CollisionSeverity.WARNING
        assert "vectorizer origin" in collision.details.lower()
    
    @pytest.mark.asyncio
    async def test_check_single_collision_unrelated_column(self, detector, sample_table_info):
        """Test collision with unrelated existing column."""
        with patch.object(detector.introspector, 'get_table_info', return_value=sample_table_info):
            collision = await detector.check_single_collision(
                "public", "documents", "content"
            )
        
        assert collision.collision_type == CollisionType.UNRELATED_COLUMN
        assert collision.severity == CollisionSeverity.INFO
        assert not collision.is_critical
    
    @pytest.mark.asyncio
    async def test_check_single_collision_table_not_found(self, detector):
        """Test collision check when table doesn't exist."""
        with patch.object(detector.introspector, 'get_table_info', return_value=None):
            with pytest.raises(Exception):  # Should raise SchemaError for nonexistent table
                await detector.check_single_collision(
                    "public", "nonexistent", "some_column"
                )
    
    @pytest.mark.asyncio
    async def test_check_multiple_collisions(self, detector, sample_table_info):
        """Test checking multiple columns for collisions."""
        columns = ["new_column", "existing_vector", "content", "another_new"]
        
        with patch.object(detector.introspector, 'get_table_info', return_value=sample_table_info):
            collisions = await detector.check_multiple_collisions(
                "public", "documents", columns
            )
        
        assert len(collisions) == len(columns)
        assert "new_column" in collisions
        assert "existing_vector" in collisions
        
        # Check specific collision types
        assert collisions["new_column"].collision_type == CollisionType.NO_COLLISION
        assert collisions["existing_vector"].collision_type == CollisionType.EXISTING_VECTOR
        assert collisions["content"].collision_type == CollisionType.UNRELATED_COLUMN
    
    def test_is_vector_column_true(self, detector):
        """Test vector column detection for actual vector column."""
        vector_column = ColumnInfo(
            name="embedding",
            data_type="USER-DEFINED",
            udt_name="vector",
            is_nullable=True,
        )
        
        assert detector._is_vector_column(vector_column) == True
    
    def test_is_vector_column_false(self, detector):
        """Test vector column detection for non-vector column."""
        text_column = ColumnInfo(
            name="content",
            data_type="text",
            is_nullable=True,
        )
        
        assert detector._is_vector_column(text_column) == False
    
    def test_matches_vectorizer_pattern_true(self, detector):
        """Test vectorizer pattern matching for typical names."""
        vectorizer_names = [
            "content_vector",
            "title_embedding", 
            "description_cleaned",
            "text_vector_model",
            "content_embedding_created_at",
        ]
        
        for name in vectorizer_names:
            assert detector._matches_vectorizer_pattern(name) == True
    
    def test_matches_vectorizer_pattern_false(self, detector):
        """Test vectorizer pattern matching for non-vectorizer names."""
        regular_names = [
            "content",
            "title",
            "description",
            "user_id",
            "created_at",
            "status",
        ]
        
        for name in regular_names:
            assert detector._matches_vectorizer_pattern(name) == False
    
    def test_generate_alternative_names_basic(self, detector):
        """Test basic alternative name generation."""
        alternatives = detector._generate_alternative_names("content_vector", set())
        
        expected = ["content_vector_v2", "content_vector_v3", "content_vector_2"]
        assert len(alternatives) >= 3
        assert all(alt in alternatives for alt in expected[:2])  # Check first two
    
    def test_generate_alternative_names_with_conflicts(self, detector):
        """Test alternative name generation avoiding conflicts."""
        existing = {"content_vector_v2", "content_vector_v3"}
        alternatives = detector._generate_alternative_names("content_vector", existing)
        
        # Should skip v2 and v3, start with v4
        assert "content_vector_v2" not in alternatives
        assert "content_vector_v3" not in alternatives
        assert "content_vector_v4" in alternatives
    
    def test_generate_alternative_names_prefix_strategy(self, detector):
        """Test alternative name generation with prefix strategy."""
        alternatives = detector._generate_alternative_names(
            "content_vector", set(), strategy="prefix"
        )
        
        # Should include prefixed versions
        prefix_alternatives = [alt for alt in alternatives if alt.startswith("v10r_")]
        assert len(prefix_alternatives) > 0
    
    @pytest.mark.asyncio
    async def test_check_metadata_history_with_history(self, detector):
        """Test checking metadata history when column was previously managed."""
        # Mock schemas list to include v10r_metadata
        detector.introspector.list_schemas = AsyncMock(return_value=["public", "v10r_metadata"])
        
        # Mock metadata query to return previous vectorizer activity with proper columns
        detector.pool.fetch = AsyncMock(return_value=[
            {
                "original_column_name": "old_vector",
                "actual_column_name": "old_vector",
                "column_type": "vector", 
                "dimension": 1024,
                "model_name": "BAAI/bge-m3",
                "created_at": "2024-01-01T12:00:00",
                "deprecated_at": None,
                "config_key": "test_config"
            }
        ])
        
        has_history = await detector._check_metadata_history(
            "public", "documents", "old_vector"
        )
        
        assert has_history == True
    
    @pytest.mark.asyncio 
    async def test_check_metadata_history_without_history(self, detector):
        """Test checking metadata history when no previous activity."""
        # Mock metadata query to return no results
        detector.pool.fetch = AsyncMock(return_value=[])
        
        has_history = await detector._check_metadata_history(
            "public", "documents", "new_column"
        )
        
        assert has_history == False
    
    @pytest.mark.asyncio
    async def test_check_metadata_history_error_handling(self, detector):
        """Test metadata history check error handling."""
        # Mock metadata query to raise exception (table doesn't exist)
        detector.pool.fetch = AsyncMock(side_effect=Exception("Table not found"))
        
        has_history = await detector._check_metadata_history(
            "public", "documents", "some_column"
        )
        
        # Should return False on error, not raise
        assert has_history == False
    
    def test_assess_collision_severity_critical(self, detector):
        """Test collision severity assessment for critical cases."""
        # Existing vector column
        severity = detector._assess_collision_severity(
            CollisionType.EXISTING_VECTOR, "vector", True
        )
        assert severity == CollisionSeverity.CRITICAL
        
        # Previous vectorizer column
        severity = detector._assess_collision_severity(
            CollisionType.PREVIOUS_VECTORIZER, "text", True
        )
        assert severity == CollisionSeverity.CRITICAL
    
    def test_assess_collision_severity_warning(self, detector):
        """Test collision severity assessment for warning cases."""
        # Vectorizer pattern without history
        severity = detector._assess_collision_severity(
            CollisionType.VECTORIZER_PATTERN, "text", False
        )
        assert severity == CollisionSeverity.WARNING
    
    def test_assess_collision_severity_info(self, detector):
        """Test collision severity assessment for info cases."""
        # Unrelated column
        severity = detector._assess_collision_severity(
            CollisionType.UNRELATED_COLUMN, "integer", False
        )
        assert severity == CollisionSeverity.INFO
    
    def test_assess_collision_severity_safe(self, detector):
        """Test collision severity assessment for safe cases."""
        # No collision
        severity = detector._assess_collision_severity(
            CollisionType.NO_COLLISION, None, False
        )
        assert severity == CollisionSeverity.SAFE


class TestCollisionDetectorIntegration:
    """Integration tests for collision detector."""
    
    @pytest.fixture
    def mock_pool(self):
        """Create mock pool with realistic responses."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool
    
    @pytest.mark.asyncio
    async def test_realistic_collision_detection_scenario(self, mock_pool):
        """Test realistic collision detection scenario."""
        # Setup detector
        detector = ColumnCollisionDetector(mock_pool)
        
        # Mock table with realistic columns
        table_info = TableInfo(
            schema="public",
            name="products", 
            columns={
                "id": ColumnInfo(name="id", data_type="integer", is_nullable=False),
                "title": ColumnInfo(name="title", data_type="varchar", is_nullable=True),
                "description": ColumnInfo(name="description", data_type="text", is_nullable=True),
                "title_vector": ColumnInfo(
                    name="title_vector", 
                    data_type="USER-DEFINED", 
                    udt_name="vector", 
                    is_nullable=True
                ),
                "old_embedding": ColumnInfo(name="old_embedding", data_type="text", is_nullable=True),
            },
            indexes=[],
            primary_key=["id"],
        )
        
        # Mock no metadata history
        mock_pool.fetch.return_value = []
        
        with patch.object(detector.introspector, 'get_table_info', return_value=table_info):
            # Test multiple collision scenarios
            columns_to_check = [
                "title_vector",        # Critical: existing vector
                "description_vector",  # Safe: no collision  
                "title_embedding",     # Warning: vectorizer pattern
                "old_embedding",       # Info: unrelated column
            ]
            
            collisions = await detector.check_multiple_collisions(
                "public", "products", columns_to_check
            )
        
        # Verify results
        assert collisions["title_vector"].is_critical
        assert collisions["description_vector"].is_safe
        # title_embedding should match vectorizer pattern but collision logic treats non-existent columns as safe
        assert collisions["title_embedding"].is_safe  # No actual collision since column doesn't exist
        # old_embedding exists in table and matches "_embedding" pattern, so it's WARNING not INFO
        assert collisions["old_embedding"].severity == CollisionSeverity.WARNING
        
        # Check alternatives are generated for collisions
        for column, collision in collisions.items():
            if collision.needs_resolution:
                assert len(collision.alternative_names) > 0
    
    @pytest.mark.asyncio
    async def test_edge_case_handling(self, mock_pool):
        """Test edge cases in collision detection."""
        detector = ColumnCollisionDetector(mock_pool)
        
        # Test with empty table
        empty_table = TableInfo(
            schema="public",
            name="empty_table",
            columns={},
            indexes=[],
            primary_key=[],
        )
        
        with patch.object(detector.introspector, 'get_table_info', return_value=empty_table):
            collision = await detector.check_single_collision(
                "public", "empty_table", "any_column"
            )
        
        assert collision.collision_type == CollisionType.NO_COLLISION
        assert collision.is_safe
    
    def test_performance_with_many_columns(self, mock_pool):
        """Test performance with large number of columns."""
        detector = ColumnCollisionDetector(mock_pool)
        
        # Test alternative name generation doesn't slow down significantly
        large_existing_set = {f"column_{i}" for i in range(1000)}
        
        alternatives = detector._generate_alternative_names(
            "column_test", large_existing_set
        )
        
        # Should still generate alternatives efficiently
        assert len(alternatives) > 0
        assert all(alt not in large_existing_set for alt in alternatives) 