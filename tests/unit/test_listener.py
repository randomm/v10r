"""
Unit tests for the PostgreSQL NOTIFY/LISTEN service.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from v10r.listener import VectorListener
from v10r.config import V10rConfig
from v10r.exceptions import ListenerError, DatabaseConnectionError

@pytest.fixture
def sample_listener_config():
    """Sample listener configuration."""
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
    
    # Tables configuration
    table_config = MagicMock()
    table_config.schema = "public"
    table_config.table = "test_table"
    table_config.text_column = "content"
    table_config.id_column = "id"
    table_config.embedding_config = "test_embedding"
    db_config.tables = [table_config]
    
    config.databases = [db_config]
    
    # Queue configuration
    config.queue = MagicMock()
    config.queue.type = "redis"
    config.queue.connection = {"host": "localhost", "port": 6379, "db": 0}
    config.queue.queue_name = "v10r:embedding_queue"
    
    # Embeddings configuration
    embedding_config = MagicMock()
    embedding_config.provider = "openai"
    embedding_config.model = "text-embedding-3-small"
    embedding_config.dimensions = 1536
    config.embeddings = {"test_embedding": embedding_config}
    
    return config

@pytest.fixture
def sample_notify_payload():
    """Sample PostgreSQL NOTIFY payload."""
    return {
        "database": "test_db",
        "schema": "public", 
        "table": "test_table",
        "operation": "INSERT",
        "id": 1
    }

class TestVectorListenerInitialization:
    """Test VectorListener initialization."""
    
    def test_listener_initialization(self, sample_listener_config):
        """Test listener initialization."""
        from v10r.listener import VectorListener
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        assert listener.config == sample_listener_config
        assert listener.channel == "v10r_events"
        assert listener.running is False
        assert listener.database_managers == {}
        assert listener.redis_pool is None
        assert listener.listeners == {}

    def test_listener_initialization_custom_channel(self, sample_listener_config):
        """Test listener initialization with custom channel."""
        from v10r.listener import VectorListener
        
        listener = VectorListener(sample_listener_config, "custom_channel")
        
        assert listener.channel == "custom_channel"

class TestVectorListenerLifecycle:
    """Test VectorListener start/stop lifecycle."""
    
    @pytest.mark.asyncio
    @patch('v10r.listener.redis.ConnectionPool')
    @patch('v10r.listener.redis.Redis')
    @patch('v10r.listener.asyncpg.connect')
    @patch('v10r.listener.DatabaseManager')
    async def test_listener_start_success(self, mock_db_manager_class, mock_connect, mock_redis_class, mock_pool_class, sample_listener_config):
        """Test successful listener startup."""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_pool_class.return_value = mock_pool
        
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        mock_conn = AsyncMock()
        mock_conn.add_listener = AsyncMock()
        mock_connect.return_value = mock_conn
        
        mock_db_manager = AsyncMock()
        mock_db_manager_class.return_value = mock_db_manager
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        # Mock _run_forever to avoid infinite loop
        listener._run_forever = AsyncMock()
        
        await listener.start()
        
        assert listener.running is True
        mock_redis.ping.assert_called_once()
        mock_conn.add_listener.assert_called_once()

    @pytest.mark.asyncio
    async def test_listener_stop(self, sample_listener_config):
        """Test listener stop."""
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        # Setup some mock connections
        mock_conn = AsyncMock()
        # is_closed() is not async in asyncpg, so use a regular Mock method
        mock_conn.is_closed = MagicMock(return_value=False)
        mock_conn.close = AsyncMock()
        listener.listeners = {"test_db": mock_conn}
        
        mock_manager = AsyncMock()
        mock_manager.close = AsyncMock()
        listener.database_managers = {"test_db": mock_manager}
        
        mock_pool = AsyncMock()
        mock_pool.disconnect = AsyncMock()
        listener.redis_pool = mock_pool
        
        listener.running = True
        
        await listener.stop()
        
        assert listener.running is False
        mock_conn.close.assert_called_once()
        mock_manager.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()

class TestNotificationHandling:
    """Test PostgreSQL notification processing."""
    
    @pytest.mark.asyncio
    @patch('v10r.listener.redis.ConnectionPool')
    @patch('v10r.listener.redis.Redis')
    async def test_handle_notification_success(self, mock_redis_class, mock_pool_class, sample_listener_config, sample_notify_payload):
        """Test successful notification handling."""
        # Setup mocks
        mock_pool = AsyncMock()
        mock_pool_class.return_value = mock_pool
        
        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        listener.redis_pool = mock_pool
        
        # Setup find_table_config to return config
        listener._find_table_config = MagicMock(return_value={
            'schema': 'public',
            'table': 'test_table',
            'text_column': 'content',
            'id_column': 'id',
            'embedding_config': 'test_embedding'
        })
        
        # Mock _queue_task
        listener._queue_task = AsyncMock()
        
        # Mock connection
        mock_conn = AsyncMock()
        
        payload_json = json.dumps(sample_notify_payload)
        
        # Call the actual handler
        await listener._handle_notification(mock_conn, 12345, "v10r_events", payload_json)
        
        # Verify task was queued
        listener._queue_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_notification_invalid_json(self, sample_listener_config):
        """Test notification with invalid JSON."""
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        mock_conn = AsyncMock()
        
        # Should not raise exception, just log warning
        await listener._handle_notification(mock_conn, 12345, "v10r_events", "invalid json")
        
        # No assertions needed - just shouldn't crash

    @pytest.mark.asyncio
    async def test_handle_notification_missing_fields(self, sample_listener_config):
        """Test notification with missing required fields."""
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        mock_conn = AsyncMock()
        
        invalid_payload = {"database": "test_db"}  # Missing required fields
        payload_json = json.dumps(invalid_payload)
        
        # Should not raise exception, just log warning
        await listener._handle_notification(mock_conn, 12345, "v10r_events", payload_json)
        
        # No assertions needed - just shouldn't crash

class TestTableConfigLookup:
    """Test table configuration lookup functionality."""
    
    def test_find_table_config_success(self, sample_listener_config):
        """Test finding table configuration successfully."""
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        config = listener._find_table_config("test_db", "public", "test_table")
        
        assert config is not None
        assert config['schema'] == "public"
        assert config['table'] == "test_table"
        assert config['text_column'] == "content"
        assert config['embedding_config'] == "test_embedding"

    def test_find_table_config_not_found(self, sample_listener_config):
        """Test table configuration not found."""
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        config = listener._find_table_config("test_db", "public", "nonexistent_table")
        
        assert config is None

class TestQueueOperations:
    """Test Redis queue operations."""
    
    @pytest.mark.asyncio
    @patch('v10r.listener.redis.Redis')
    async def test_queue_task_success(self, mock_redis_class, sample_listener_config):
        """Test successful task queuing."""
        mock_redis = AsyncMock()
        mock_redis.lpush = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        mock_pool = AsyncMock()
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        listener.redis_pool = mock_pool
        
        task = {
            'event': {'database': 'test_db', 'table': 'test_table'},
            'table_config': {'text_column': 'content'},
            'timestamp': 1234567890,
            'retry_count': 0
        }
        
        await listener._queue_task(task)
        
        # Verify Redis connection was created and lpush was called
        mock_redis_class.assert_called_once_with(connection_pool=mock_pool)
        mock_redis.lpush.assert_called_once()

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    @patch('v10r.listener.redis.ConnectionPool')
    @patch('v10r.listener.redis.Redis')
    async def test_redis_connection_failure(self, mock_redis_class, mock_pool_class, sample_listener_config):
        """Test Redis connection failure handling."""
        # Setup Redis to fail
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")
        mock_redis_class.return_value = mock_redis
        
        mock_pool = AsyncMock()
        mock_pool_class.return_value = mock_pool
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        with pytest.raises(ListenerError):
            await listener.start()

    @pytest.mark.asyncio
    @patch('v10r.listener.redis.ConnectionPool')
    @patch('v10r.listener.redis.Redis')
    @patch('v10r.listener.asyncpg.connect')
    async def test_database_connection_failure(self, mock_connect, mock_redis_class, mock_pool_class, sample_listener_config):
        """Test database connection failure handling."""
        # Setup Redis to succeed
        mock_pool = AsyncMock()
        mock_pool_class.return_value = mock_pool
        
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis_class.return_value = mock_redis
        
        # Setup database connection to fail
        mock_connect.side_effect = Exception("Database connection failed")
        
        listener = VectorListener(sample_listener_config, "v10r_events")
        
        with pytest.raises(ListenerError):
            await listener.start() 