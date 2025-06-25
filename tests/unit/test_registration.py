"""
Unit tests for the table registration service.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import asynccontextmanager

from v10r.registration import TableRegistrationService, RegistrationResult
from v10r.config import V10rConfig
from v10r.exceptions import RegistrationError, SchemaError, ValidationError
from v10r.schema.collision_detector import CollisionSeverity

@pytest.fixture
def sample_registration_config():
    """Sample registration configuration."""
    config = MagicMock(spec=V10rConfig)
    
    # Create proper database config with connection object
    db_config = MagicMock()
    db_config.name = "test_db"
    db_config.connection = MagicMock()
    db_config.connection.host = "localhost"
    db_config.connection.port = 5432
    db_config.connection.database = "test"
    db_config.connection.user = "user"
    db_config.connection.password = "pass"
    config.databases = [db_config]
    
    # Embeddings configuration
    embedding_config = MagicMock()
    embedding_config.provider = "openai"
    embedding_config.model = "text-embedding-3-small"
    embedding_config.dimensions = 1536
    embedding_config.api_key = "test-key"
    config.embeddings = {"test_embedding": embedding_config}
    
    return config

@pytest.fixture
def sample_registration_request():
    """Sample table registration request."""
    return {
        "database": "test_db",
        "schema": "public",
        "table": "test_table",
        "text_column": "content",
        "embedding_config": "test_embedding"
    }

@pytest.fixture
def mock_table_info():
    """Mock table info object."""
    table_info = MagicMock()
    table_info.has_column.side_effect = lambda col: col in ["id", "content", "created_at"]
    table_info.columns = ["id", "content", "created_at"]
    return table_info

class TestTableRegistrationServiceInitialization:
    """Test TableRegistrationService initialization."""
    
    def test_service_initialization(self, sample_registration_config):
        """Test service initialization."""
        service = TableRegistrationService(sample_registration_config)
        
        assert service.config == sample_registration_config
        assert service.database_managers == {}

class TestTableRegistration:
    """Test table registration functionality."""
    
    @pytest.mark.asyncio
    @patch('v10r.registration.DatabaseManager')
    @patch('v10r.registration.SchemaIntrospector')
    @patch('v10r.registration.SafeSchemaOperations')
    @patch('v10r.registration.ColumnCollisionDetector')
    async def test_register_table_success(self, mock_collision_detector_class, mock_operations_class,
                                        mock_introspector_class, mock_db_manager_class, 
                                        sample_registration_config, mock_table_info):
        """Test successful table registration."""
        # Setup mocks
        mock_db_manager = AsyncMock()
        mock_db_manager.pool = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager
        
        mock_introspector = AsyncMock()
        mock_introspector.table_exists.return_value = True
        mock_introspector.get_table_info.return_value = mock_table_info
        mock_introspector_class.return_value = mock_introspector
        
        mock_operations = AsyncMock()
        mock_operations.create_vector_column.return_value = MagicMock()
        mock_operations.add_column_if_not_exists.return_value = MagicMock()
        mock_operations_class.return_value = mock_operations
        
        mock_collision_detector = AsyncMock()
        mock_collision_detector_class.return_value = mock_collision_detector
        
        service = TableRegistrationService(sample_registration_config)
        service._create_vectorization_trigger = AsyncMock()
        service._resolve_column_collision = AsyncMock(side_effect=lambda *args: args[2])  # Return original name
        
        result = await service.register_table(
            database="test_db",
            schema="public", 
            table="test_table",
            text_column="content",
            embedding_config="test_embedding"
        )
        
        assert result.success is True
        assert result.vector_column == "content_vector"
        assert result.model_column == "content_embedding_model"
        assert result.timestamp_column == "content_vector_created_at"

    @pytest.mark.asyncio
    @patch('v10r.registration.DatabaseManager')
    @patch('v10r.registration.SchemaIntrospector')
    async def test_register_table_table_not_exists(self, mock_introspector_class, mock_db_manager_class,
                                                  sample_registration_config):
        """Test table registration when table doesn't exist."""
        # Setup mocks
        mock_db_manager = AsyncMock()
        mock_db_manager.pool = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager
        
        mock_introspector = AsyncMock()
        mock_introspector.table_exists.return_value = False
        mock_introspector_class.return_value = mock_introspector
        
        service = TableRegistrationService(sample_registration_config)
        
        result = await service.register_table(
            database="test_db",
            schema="public",
            table="nonexistent_table", 
            text_column="content",
            embedding_config="test_embedding"
        )
        
        assert result.success is False
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    @patch('v10r.registration.DatabaseManager')
    @patch('v10r.registration.SchemaIntrospector')
    async def test_register_table_text_column_not_exists(self, mock_introspector_class, mock_db_manager_class,
                                                        sample_registration_config, mock_table_info):
        """Test table registration when text column doesn't exist."""
        # Setup mocks
        mock_db_manager = AsyncMock()
        mock_db_manager.pool = MagicMock()
        mock_db_manager_class.return_value = mock_db_manager
        
        mock_introspector = AsyncMock()
        mock_introspector.table_exists.return_value = True
        mock_introspector.get_table_info.return_value = mock_table_info
        mock_introspector_class.return_value = mock_introspector
        
        # Mock table info to not have the text column
        mock_table_info.has_column.side_effect = lambda col: col != "nonexistent_column"
        
        service = TableRegistrationService(sample_registration_config)
        
        result = await service.register_table(
            database="test_db",
            schema="public",
            table="test_table",
            text_column="nonexistent_column",
            embedding_config="test_embedding"
        )
        
        assert result.success is False
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_register_table_invalid_embedding_config(self, sample_registration_config):
        """Test table registration with invalid embedding config."""
        service = TableRegistrationService(sample_registration_config)
        
        result = await service.register_table(
            database="test_db",
            schema="public",
            table="test_table", 
            text_column="content",
            embedding_config="nonexistent_embedding"
        )
        
        assert result.success is False
        assert "not found" in result.error

class TestDatabaseManager:
    """Test database manager operations."""
    
    @pytest.mark.asyncio
    @patch('v10r.registration.DatabaseManager')
    async def test_get_database_manager_new(self, mock_db_manager_class, sample_registration_config):
        """Test getting a new database manager."""
        mock_db_manager = AsyncMock()
        mock_db_manager_class.return_value = mock_db_manager
        
        service = TableRegistrationService(sample_registration_config)
        
        manager = await service._get_database_manager("test_db")
        
        assert manager == mock_db_manager
        assert service.database_managers["test_db"] == mock_db_manager

    @pytest.mark.asyncio
    async def test_get_database_manager_existing(self, sample_registration_config):
        """Test getting an existing database manager."""
        service = TableRegistrationService(sample_registration_config)
        
        # Setup existing manager
        existing_manager = AsyncMock()
        service.database_managers["test_db"] = existing_manager
        
        manager = await service._get_database_manager("test_db")
        
        assert manager == existing_manager

    @pytest.mark.asyncio
    async def test_get_database_manager_invalid_database(self, sample_registration_config):
        """Test getting database manager for invalid database."""
        service = TableRegistrationService(sample_registration_config)

        with pytest.raises(ValidationError):
            await service._get_database_manager("nonexistent_db")

class TestColumnCollisionResolution:
    """Test column collision resolution."""
    
    @pytest.mark.asyncio
    async def test_resolve_column_collision_no_collision(self, sample_registration_config, mock_table_info):
        """Test column collision resolution when no collision exists."""
        service = TableRegistrationService(sample_registration_config)
        
        # Mock collision detector
        mock_collision_detector = AsyncMock()
        mock_collision_detector.check_column_collision.return_value = []
        
        warnings = []
        result = await service._resolve_column_collision(
            mock_collision_detector,
            mock_table_info,
            "new_vector_column",
            "vector",
            warnings
        )
        
        assert result == "new_vector_column"
        assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_resolve_column_collision_with_collision(self, sample_registration_config, mock_table_info):
        """Test column collision resolution when collision exists."""
        service = TableRegistrationService(sample_registration_config)
        
        # Mock table_info to show column exists (causing collision)
        mock_table_info.has_column.side_effect = lambda col: col == "new_vector_column"  # Only this column exists
        mock_table_info.schema = "public"
        mock_table_info.name = "test_table"
        
        # Mock collision detector to return collision info with non-critical severity
        mock_collision_detector = AsyncMock()
        mock_collision_info = MagicMock()
        mock_collision_info.severity = CollisionSeverity.WARNING  # Not critical, so alternative will be used
        mock_collision_info.alternative_names = ["new_vector_column_v2"]
        mock_collision_detector.check_single_collision.return_value = mock_collision_info
        
        warnings = []
        result = await service._resolve_column_collision(
            mock_collision_detector,
            mock_table_info,
            "new_vector_column",
            "vector",
            warnings
        )
        
        assert result == "new_vector_column_v2"
        assert len(warnings) > 0

class TestTriggerCreation:
    """Test vectorization trigger creation."""
    
    @pytest.mark.asyncio
    async def test_create_vectorization_trigger_success(self, sample_registration_config):
        """Test successful trigger creation."""
        service = TableRegistrationService(sample_registration_config)
        
        # Mock database manager with proper async context manager
        mock_db_manager = AsyncMock()
        mock_conn = AsyncMock()
        mock_conn.execute.return_value = None
        
        # Mock the pool.acquire() context manager properly
        @asynccontextmanager
        async def mock_acquire():
            yield mock_conn
            
        mock_pool = AsyncMock()
        mock_pool.acquire = mock_acquire
        mock_db_manager.pool = mock_pool
        
        await service._create_vectorization_trigger(
            mock_db_manager,
            "public",
            "test_table",
            "content",
            "content_vector"
        )
        
        # Verify trigger SQL was executed
        mock_conn.execute.assert_called_once()
        
        # Get the executed SQL
        call_args = mock_conn.execute.call_args
        sql = call_args[0][0]
        
        assert "CREATE OR REPLACE TRIGGER" in sql
        assert "test_table_content_vector_trigger" in sql
        assert "v10r.generic_vector_notify" in sql

class TestValidation:
    """Test validation methods."""
    
    @pytest.mark.asyncio
    async def test_validate_registration_request_valid(self, sample_registration_config):
        """Test validation of valid registration request."""
        service = TableRegistrationService(sample_registration_config)
        
        # Should not raise exception for valid request
        await service._validate_registration_request(
            "test_db",
            "public", 
            "test_table",
            "content",
            "test_embedding"
        )

    @pytest.mark.asyncio
    async def test_validate_registration_request_invalid_database(self, sample_registration_config):
        """Test validation with invalid database."""
        service = TableRegistrationService(sample_registration_config)
        
        with pytest.raises(ValidationError):
            await service._validate_registration_request(
                "",  # Empty database name
                "public",
                "test_table", 
                "content",
                "test_embedding"
            )

    @pytest.mark.asyncio
    async def test_validate_registration_request_invalid_schema(self, sample_registration_config):
        """Test validation with invalid schema."""
        service = TableRegistrationService(sample_registration_config)
        
        with pytest.raises(ValidationError):
            await service._validate_registration_request(
                "test_db",
                "",  # Empty schema name
                "test_table",
                "content", 
                "test_embedding"
            )

    @pytest.mark.asyncio
    async def test_validate_registration_request_invalid_table(self, sample_registration_config):
        """Test validation with invalid table."""
        service = TableRegistrationService(sample_registration_config)
        
        with pytest.raises(ValidationError):
            await service._validate_registration_request(
                "test_db",
                "public",
                "",  # Empty table name
                "content",
                "test_embedding"
            )

    @pytest.mark.asyncio
    async def test_validate_registration_request_invalid_text_column(self, sample_registration_config):
        """Test validation with invalid text column."""
        service = TableRegistrationService(sample_registration_config)
        
        with pytest.raises(ValidationError):
            await service._validate_registration_request(
                "test_db",
                "public",
                "test_table",
                "",  # Empty text column name
                "test_embedding"
            )

    @pytest.mark.asyncio
    async def test_validate_registration_request_invalid_embedding_config(self, sample_registration_config):
        """Test validation with invalid embedding config."""
        service = TableRegistrationService(sample_registration_config)
        
        with pytest.raises(ValidationError):
            await service._validate_registration_request(
                "test_db",
                "public",
                "test_table",
                "content",
                ""  # Empty embedding config name
            )

class TestCleanup:
    """Test service cleanup."""
    
    @pytest.mark.asyncio
    async def test_close_service(self, sample_registration_config):
        """Test closing the service and cleaning up resources."""
        service = TableRegistrationService(sample_registration_config)
        
        # Setup mock database managers
        mock_manager1 = AsyncMock()
        mock_manager2 = AsyncMock()
        service.database_managers = {
            "db1": mock_manager1,
            "db2": mock_manager2
        }
        
        await service.close()
        
        # Verify all managers were closed
        mock_manager1.close.assert_called_once()
        mock_manager2.close.assert_called_once()

class TestRegistrationResult:
    """Test RegistrationResult dataclass."""
    
    def test_registration_result_success(self):
        """Test RegistrationResult for successful registration."""
        result = RegistrationResult(
            success=True,
            vector_column="content_vector",
            model_column="content_model",
            timestamp_column="content_created_at",
            warnings=["Warning message"]
        )
        
        assert result.success is True
        assert result.vector_column == "content_vector"
        assert result.model_column == "content_model" 
        assert result.timestamp_column == "content_created_at"
        assert len(result.warnings) == 1
        assert result.error is None

    def test_registration_result_failure(self):
        """Test RegistrationResult for failed registration."""
        result = RegistrationResult(
            success=False,
            error="Registration failed"
        )
        
        assert result.success is False
        assert result.error == "Registration failed"
        assert result.vector_column is None
        assert result.model_column is None
        assert result.timestamp_column is None
        assert len(result.warnings) == 0  # Should be initialized as empty list

    def test_registration_result_warnings_initialization(self):
        """Test that warnings list is properly initialized."""
        result = RegistrationResult(success=True)
        
        assert result.warnings == []
        assert isinstance(result.warnings, list) 