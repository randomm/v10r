"""
Tests for v10r.worker module.

Tests the VectorizerWorker class and related functionality for processing
vectorization tasks from Redis queues.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Dict, List, Optional, Any

from v10r.worker import VectorizerWorker
from v10r.config import V10rConfig
from v10r.exceptions import WorkerError, EmbeddingError, DatabaseError


class TestVectorizerWorker:
    """Test VectorizerWorker class."""

    @pytest.fixture
    def mock_config(self):
        """Mock V10rConfig for testing."""
        config = MagicMock(spec=V10rConfig)
        
        # Mock queue configuration
        config.queue = MagicMock()
        config.queue.connection = {
            'host': 'localhost',
            'port': 6379,
            'db': 0
        }
        config.queue.name = 'v10r_tasks'
        config.queue.max_retries = 3
        config.queue.retry_delay = 60
        
        # Mock databases configuration
        config.databases = []
        mock_db = MagicMock()
        mock_db.name = 'test_db'
        mock_db.connection = MagicMock()
        mock_db.connection.host = 'localhost'
        mock_db.connection.port = 5432
        mock_db.connection.database = 'test_db'
        mock_db.connection.user = 'test_user'
        mock_db.connection.password = 'test_pass'
        config.databases.append(mock_db)
        
        # Mock embeddings configuration
        config.embeddings = {}
        mock_embedding = MagicMock()
        mock_embedding.provider = 'openai'
        mock_embedding.model = 'text-embedding-3-small'
        mock_embedding.dimensions = 1536
        config.embeddings['test_embedding'] = mock_embedding
        
        # Mock workers configuration
        config.workers = MagicMock()
        config.workers.concurrency = 4
        config.workers.batch_timeout = 60
        
        # Mock cleaning configs
        config.cleaning_configs = {}
        
        return config

    @pytest.fixture
    def worker(self, mock_config):
        """VectorizerWorker instance with mocked config."""
        return VectorizerWorker(mock_config)

    def test_init(self, worker, mock_config):
        """Test VectorizerWorker initialization."""
        assert worker.config == mock_config
        assert worker.redis_pool is None
        assert worker.database_managers == {}
        assert worker.embedding_clients == {}
        assert worker.preprocessing_pipelines == {}
        assert worker.running is False
        assert worker.worker_id.startswith("worker-")

    @pytest.mark.asyncio
    async def test_setup_redis_success(self, worker):
        """Test successful Redis setup."""
        with patch('redis.asyncio.ConnectionPool') as mock_pool_class:
            with patch('redis.asyncio.Redis') as mock_redis_class:
                mock_pool = MagicMock()
                mock_pool_class.return_value = mock_pool
                
                mock_redis = AsyncMock()
                mock_redis_class.return_value = mock_redis
                mock_redis.ping.return_value = True
                
                await worker._setup_redis()
                
                assert worker.redis_pool == mock_pool
                mock_pool_class.assert_called_once_with(
                    host='localhost',
                    port=6379,
                    db=0,
                    decode_responses=False
                )
                mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_redis_connection_error(self, worker):
        """Test Redis setup with connection error."""
        with patch('redis.asyncio.ConnectionPool') as mock_pool_class:
            with patch('redis.asyncio.Redis') as mock_redis_class:
                mock_redis = AsyncMock()
                mock_redis_class.return_value = mock_redis
                mock_redis.ping.side_effect = Exception("Connection failed")
                
                with pytest.raises(WorkerError) as exc_info:
                    await worker._setup_redis()
                
                assert "Failed to connect to Redis" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_setup_database_managers_success(self, worker):
        """Test successful database manager setup."""
        with patch('v10r.worker.DatabaseManager') as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager
            
            await worker._setup_database_managers()
            
            assert len(worker.database_managers) == 1
            assert "test_db" in worker.database_managers
            assert worker.database_managers["test_db"] == mock_manager
            mock_manager_class.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_database_managers_error(self, worker):
        """Test database manager setup with error."""
        with patch('v10r.worker.DatabaseManager') as mock_manager_class:
            mock_manager_class.side_effect = Exception("DB connection failed")
            
            with pytest.raises(DatabaseError) as exc_info:
                await worker._setup_database_managers()
            
            assert "Database manager setup failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_setup_embedding_clients_success(self, worker):
        """Test successful embedding client setup."""
        with patch('v10r.worker.EmbeddingClientFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            
            mock_client = MagicMock()  # Not AsyncMock - create_client is async but returns regular object
            mock_factory.create_client = AsyncMock(return_value=mock_client)
            
            await worker._setup_embedding_clients()
            
            assert len(worker.embedding_clients) == 1
            assert "test_embedding" in worker.embedding_clients
            assert worker.embedding_clients["test_embedding"] == mock_client
            mock_factory.create_client.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_embedding_clients_error(self, worker):
        """Test embedding client setup error."""
        with patch('v10r.worker.EmbeddingClientFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            
            mock_factory.create_client = AsyncMock(side_effect=Exception("Client setup failed"))
            
            with pytest.raises(EmbeddingError):
                await worker._setup_embedding_clients()

    @pytest.mark.asyncio
    async def test_setup_preprocessing_pipelines_success(self, worker):
        """Test successful preprocessing pipeline setup."""
        # Add a cleaning config to the mock config
        mock_cleaning_config = MagicMock()
        worker.config.cleaning_configs['test_cleaning'] = mock_cleaning_config
        
        with patch('v10r.worker.PreprocessingFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            
            mock_pipeline = MagicMock()
            mock_factory.create_pipeline.return_value = mock_pipeline
            
            mock_basic_pipeline = MagicMock()
            mock_factory.get_basic_pipeline.return_value = mock_basic_pipeline
            
            await worker._setup_preprocessing_pipelines()
            
            assert len(worker.preprocessing_pipelines) >= 1
            assert "test_cleaning" in worker.preprocessing_pipelines
            mock_factory.create_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_preprocessing_pipelines_with_fallback(self, worker):
        """Test preprocessing pipeline setup with fallback to basic."""
        with patch('v10r.worker.PreprocessingFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            
            mock_basic_pipeline = MagicMock()
            mock_factory.get_basic_pipeline.return_value = mock_basic_pipeline
            
            # No cleaning configs in worker.config.cleaning_configs
            await worker._setup_preprocessing_pipelines()
            
            assert len(worker.preprocessing_pipelines) >= 1
            assert "basic_cleanup" in worker.preprocessing_pipelines
            mock_factory.get_basic_pipeline.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_success(self, worker):
        """Test successful worker stop."""
        # Setup some resources to close
        mock_client = AsyncMock()
        worker.embedding_clients["test_embedding"] = mock_client
        
        mock_manager = AsyncMock()
        worker.database_managers["test_db"] = mock_manager
        
        mock_redis_pool = AsyncMock()
        worker.redis_pool = mock_redis_pool
        
        worker.running = True
        
        await worker.stop()
        
        assert worker.running is False
        mock_client.close.assert_called_once()
        mock_manager.close.assert_called_once()
        mock_redis_pool.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_errors(self, worker):
        """Test worker stop with cleanup errors."""
        # Setup resources that will fail to close
        mock_client = AsyncMock()
        mock_client.close.side_effect = Exception("Close failed")
        worker.embedding_clients["test_embedding"] = mock_client
        
        mock_manager = AsyncMock()
        mock_manager.close.side_effect = Exception("Close failed")
        worker.database_managers["test_db"] = mock_manager
        
        mock_redis_pool = AsyncMock()
        mock_redis_pool.disconnect.side_effect = Exception("Disconnect failed")
        worker.redis_pool = mock_redis_pool
        
        # Should not raise exception, just log warnings
        await worker.stop()
        assert worker.running is False

    @pytest.mark.asyncio
    async def test_get_task_success(self, worker):
        """Test successful task retrieval from Redis."""
        mock_redis_pool = AsyncMock()
        worker.redis_pool = mock_redis_pool
        
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            
            # Mock task data
            task_data = {"table": "test_db", "event": {"id": 1}}
            mock_redis.brpop.return_value = ("v10r_tasks", json.dumps(task_data).encode())
            
            result = await worker._get_task()
            
            assert result == task_data
            mock_redis.brpop.assert_called_once_with("v10r_tasks", timeout=1)

    @pytest.mark.asyncio
    async def test_get_task_timeout(self, worker):
        """Test task retrieval timeout."""
        mock_redis_pool = AsyncMock()
        worker.redis_pool = mock_redis_pool
        
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
        
            # Mock timeout (no task available)
            mock_redis.brpop.return_value = None
        
            result = await worker._get_task()
        
            assert result is None

    @pytest.mark.asyncio
    async def test_get_task_invalid_json(self, worker):
        """Test task retrieval with invalid JSON."""
        mock_redis_pool = AsyncMock()
        worker.redis_pool = mock_redis_pool
        
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.brpop.return_value = ("v10r_tasks", b"invalid json")
            
            result = await worker._get_task()
            
            assert result is None

    @pytest.mark.asyncio
    async def test_extract_text_from_event_success(self, worker):
        """Test successful text extraction from event."""
        event = {
            'id': {
                'content': 'This is test content',
                'title': 'Test Title'
            }
        }
        table_config = {
            'text_column': 'content'
        }
        
        text = worker._extract_text_from_event(event, table_config)
        assert text == 'This is test content'

    @pytest.mark.asyncio
    async def test_extract_text_from_event_missing_column(self, worker):
        """Test text extraction when column is missing."""
        event = {
            'id': {
                'title': 'Test Title'
            }
        }
        table_config = {
            'text_column': 'content'  # Missing from event data
        }
        
        text = worker._extract_text_from_event(event, table_config)
        assert text is None

    @pytest.mark.asyncio
    async def test_extract_text_from_event_empty_text(self, worker):
        """Test text extraction with empty text."""
        event = {
            'id': {
                'content': '',
                'title': 'Test Title'
            }
        }
        table_config = {
            'text_column': 'content'
        }
        
        text = worker._extract_text_from_event(event, table_config)
        assert text is None

    @pytest.mark.asyncio
    async def test_extract_text_from_event_exception(self, worker):
        """Test text extraction with malformed event data."""
        event = {
            'id': None  # This will cause an error
        }
        table_config = {
            'text_column': 'content'
        }
        
        text = worker._extract_text_from_event(event, table_config)
        assert text is None

    @pytest.mark.asyncio
    async def test_preprocess_text_success(self, worker):
        """Test successful text preprocessing."""
        # Setup preprocessing pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(return_value="cleaned text")
        worker.preprocessing_pipelines["test_cleaning"] = mock_pipeline
        
        table_config = {
            "preprocessing": {
                "enabled": True,
                "cleaning_config": "test_cleaning"
            }
        }
        
        result = await worker._preprocess_text("raw text", table_config, "worker-1")
        
        assert result == "cleaned text"
        mock_pipeline.process.assert_called_once_with("raw text")

    @pytest.mark.asyncio
    async def test_preprocess_text_no_pipeline(self, worker):
        """Test text preprocessing without pipeline configured."""
        table_config = {
            "preprocessing": {
                "enabled": False
            }
        }
        
        result = await worker._preprocess_text("raw text", table_config, "worker-1")
        
        assert result == "raw text"

    @pytest.mark.asyncio
    async def test_preprocess_text_pipeline_error(self, worker):
        """Test text preprocessing with pipeline error."""
        # Setup preprocessing pipeline that fails
        mock_pipeline = MagicMock()
        mock_pipeline.process = AsyncMock(side_effect=Exception("Processing failed"))
        worker.preprocessing_pipelines["test_cleaning"] = mock_pipeline
        
        table_config = {
            "preprocessing": {
                "enabled": True,
                "cleaning_config": "test_cleaning"
            }
        }
        
        result = await worker._preprocess_text("raw text", table_config, "worker-1")
        
        # Should return original text on error
        assert result == "raw text"

    @pytest.mark.asyncio
    async def test_worker_loop_keyboard_interrupt(self, worker):
        """Test worker loop handling keyboard interrupt."""
        worker.running = True
        
        # Mock _get_task to raise KeyboardInterrupt after first call
        call_count = 0
        async def mock_get_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise KeyboardInterrupt()
                return None
        
        with patch.object(worker, '_get_task', side_effect=mock_get_task):
            # Should exit gracefully on KeyboardInterrupt
            await worker._worker_loop("test-worker")
        
            assert call_count == 1

    @pytest.mark.asyncio
    async def test_run_workers_with_concurrency(self, worker):
        """Test running workers with configured concurrency."""
        worker.running = True
        
        # Mock _worker_loop to complete quickly
        async def mock_worker_loop(worker_name):
            worker.running = False  # Stop after first iteration
        
        with patch.object(worker, '_worker_loop', side_effect=mock_worker_loop) as mock_loop:
            with patch('asyncio.create_task') as mock_create_task:
                # Mock tasks that complete immediately
                mock_task1 = AsyncMock()
                mock_task2 = AsyncMock()
                mock_create_task.side_effect = [mock_task1, mock_task2]
                
                # Mock gather to return immediately
                with patch('asyncio.gather') as mock_gather:
                    mock_gather.return_value = None
                    
                    await worker._run_workers()
                    
                    # Should create 4 tasks (concurrency = 4)
                    assert mock_create_task.call_count == 4
                    mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_success(self, worker):
        """Test successful worker start."""
        with patch.object(worker, '_setup_redis') as mock_setup_redis:
            with patch.object(worker, '_setup_database_managers') as mock_setup_db:
                with patch.object(worker, '_setup_embedding_clients') as mock_setup_emb:
                    with patch.object(worker, '_setup_preprocessing_pipelines') as mock_setup_prep:
                        with patch.object(worker, '_run_workers') as mock_run_workers:
                            
                            await worker.start()
                            
                            assert worker.running is True
                            mock_setup_redis.assert_called_once()
                            mock_setup_db.assert_called_once()
                            mock_setup_emb.assert_called_once()
                            mock_setup_prep.assert_called_once()
                            mock_run_workers.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_setup_failure(self, worker):
        """Test start method with setup failure."""
        with patch.object(worker, '_setup_redis', side_effect=WorkerError("Redis failed")):
            with pytest.raises(WorkerError):
                await worker.start()
        
        assert worker.running is False

    @pytest.mark.asyncio
    async def test_process_task_success(self, worker):
        """Test successful task processing."""
        # Setup mock clients and managers
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        worker.embedding_clients['test_embedding'] = mock_embedding_client
        
        mock_db_manager = AsyncMock()
        mock_conn = AsyncMock()
        mock_db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        worker.database_managers['test_db'] = mock_db_manager
        
        # Setup task data
        task = {
            'event': {
                'id': {'id': 123, 'content': 'test content'}
            },
            'table_config': {
                'schema': 'public',
                'table': 'documents',
                'text_column': 'content',
                'vector_column': 'content_vector',
                'model_column': 'model',
                'id_column': 'id',
                'database_name': 'test_db',
                'embedding_config': 'test_embedding'
            }
        }
        
        # Mock methods
        with patch.object(worker, '_extract_text_from_event', return_value='test content'):
            with patch.object(worker, '_preprocess_text', return_value='preprocessed content'):
                with patch.object(worker, '_update_vector_in_database') as mock_update:
                    await worker._process_task(task, 'test-worker')
                    
                    mock_embedding_client.embed_texts.assert_called_once_with(['preprocessed content'])
                    mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_task_no_text(self, worker):
        """Test task processing with no text content."""
        task = {
            'event': {'id': {'id': 123}},
            'table_config': {'text_column': 'content'}
        }
        
        with patch.object(worker, '_extract_text_from_event', return_value=None):
            await worker._process_task(task, 'test-worker')
            # Should return early without processing

    @pytest.mark.asyncio
    async def test_process_task_no_text_after_preprocessing(self, worker):
        """Test task processing where preprocessing removes all text."""
        task = {
            'event': {'id': {'id': 123}},
            'table_config': {'text_column': 'content'}
        }
        
        with patch.object(worker, '_extract_text_from_event', return_value='some text'):
            with patch.object(worker, '_preprocess_text', return_value=''):
                await worker._process_task(task, 'test-worker')
                # Should return early after preprocessing

    @pytest.mark.asyncio
    async def test_process_task_no_embedding_client(self, worker):
        """Test task processing with missing embedding client."""
        task = {
            'event': {'id': {'id': 123}},
            'table_config': {
                'text_column': 'content',
                'embedding_config': 'missing_config'
            }
        }
        
        with patch.object(worker, '_extract_text_from_event', return_value='test content'):
            with patch.object(worker, '_preprocess_text', return_value='processed content'):
                await worker._process_task(task, 'test-worker')
                # Should return early due to missing client

    @pytest.mark.asyncio
    async def test_process_task_with_retry(self, worker):
        """Test task processing that triggers retry logic."""
        worker.config.queue.max_retries = 3
        
        task = {
            'event': {'id': {'id': 123}},
            'table_config': {
                'text_column': 'content',
                'embedding_config': 'test_embedding'
            },
            'retry_count': 1
        }
        
        # Setup mock client that will fail
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_texts.side_effect = Exception("Embedding failed")
        worker.embedding_clients['test_embedding'] = mock_embedding_client
        
        with patch.object(worker, '_extract_text_from_event', return_value='test content'):
            with patch.object(worker, '_preprocess_text', return_value='processed content'):
                with patch.object(worker, '_retry_task') as mock_retry:
                    await worker._process_task(task, 'test-worker')
                    
                    mock_retry.assert_called_once_with(task, 'test-worker')

    @pytest.mark.asyncio
    async def test_process_task_max_retries_exceeded(self, worker):
        """Test task processing that exceeds max retries."""
        worker.config.queue.max_retries = 2
        
        task = {
            'event': {'id': {'id': 123}},
            'table_config': {
                'text_column': 'content',
                'embedding_config': 'test_embedding'
            },
            'retry_count': 2  # Already at max retries
        }
        
        # Setup mock client that will fail
        mock_embedding_client = AsyncMock()
        mock_embedding_client.embed_texts.side_effect = Exception("Embedding failed")
        worker.embedding_clients['test_embedding'] = mock_embedding_client
        
        with patch.object(worker, '_extract_text_from_event', return_value='test content'):
            with patch.object(worker, '_preprocess_text', return_value='processed content'):
                with patch.object(worker, '_retry_task') as mock_retry:
                    await worker._process_task(task, 'test-worker')
                    
                    # Should not retry when max retries exceeded
                    mock_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_vector_in_database_success(self, worker):
        """Test successful database vector update."""
        # Setup mock database manager
        mock_conn = AsyncMock()
        mock_db_manager = AsyncMock()
        mock_db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        worker.database_managers['test_db'] = mock_db_manager
        
        # Setup embedding config
        mock_embedding_config = MagicMock()
        mock_embedding_config.model = 'text-embedding-3-small'
        worker.config.embeddings = {'test_embedding': mock_embedding_config}
        
        event = {
            'id': {'id': 123, 'content': 'test content'}
        }
        
        table_config = {
            'database_name': 'test_db',
            'schema': 'public',
            'table': 'documents',
            'id_column': 'id',
            'vector_column': 'content_vector',
            'model_column': 'model',
            'embedding_config': 'test_embedding'
        }
        
        embedding = [0.1, 0.2, 0.3]
        
        await worker._update_vector_in_database(event, table_config, embedding, 'test-worker')
        
        # Verify SQL was executed
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert 'UPDATE public.documents' in call_args[0]
        assert call_args[1] == embedding
        assert call_args[2] == 'text-embedding-3-small'
        assert call_args[3] == 123

    @pytest.mark.asyncio
    async def test_update_vector_in_database_no_manager(self, worker):
        """Test database update with missing database manager."""
        event = {'id': {'id': 123}}
        table_config = {'database_name': 'missing_db'}
        embedding = [0.1, 0.2, 0.3]
        
        await worker._update_vector_in_database(event, table_config, embedding, 'test-worker')
        # Should return early without error

    @pytest.mark.asyncio
    async def test_update_vector_in_database_no_row_id(self, worker):
        """Test database update with missing row ID."""
        mock_db_manager = AsyncMock()
        worker.database_managers['test_db'] = mock_db_manager
        
        event = {'id': {}}  # No ID in event data
        table_config = {
            'database_name': 'test_db',
            'id_column': 'id'
        }
        embedding = [0.1, 0.2, 0.3]
        
        await worker._update_vector_in_database(event, table_config, embedding, 'test-worker')
        # Should return early without executing SQL

    @pytest.mark.asyncio
    async def test_update_vector_in_database_unknown_embedding_config(self, worker):
        """Test database update with unknown embedding config."""
        mock_conn = AsyncMock()
        mock_db_manager = AsyncMock()
        mock_db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        worker.database_managers['test_db'] = mock_db_manager
        
        event = {'id': {'id': 123}}
        table_config = {
            'database_name': 'test_db',
            'schema': 'public',
            'table': 'documents',
            'id_column': 'id',
            'vector_column': 'content_vector',
            'model_column': 'model',
            'embedding_config': 'unknown_config'
        }
        embedding = [0.1, 0.2, 0.3]
        
        await worker._update_vector_in_database(event, table_config, embedding, 'test-worker')
        
        # Should still execute with 'unknown' model name
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert call_args[2] == 'unknown'

    @pytest.mark.asyncio
    async def test_update_vector_in_database_execution_error(self, worker):
        """Test database update with SQL execution error."""
        mock_conn = AsyncMock()
        mock_conn.execute.side_effect = Exception("SQL execution failed")
        mock_db_manager = AsyncMock()
        mock_db_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        worker.database_managers['test_db'] = mock_db_manager
        
        worker.config.embeddings = {'test_embedding': MagicMock(model='test-model')}
        
        event = {'id': {'id': 123}}
        table_config = {
            'database_name': 'test_db',
            'schema': 'public',
            'table': 'documents',
            'id_column': 'id',
            'vector_column': 'content_vector',
            'model_column': 'model',
            'embedding_config': 'test_embedding'
        }
        embedding = [0.1, 0.2, 0.3]
        
        with pytest.raises(Exception):
            await worker._update_vector_in_database(event, table_config, embedding, 'test-worker')

    @pytest.mark.asyncio
    async def test_retry_task_success(self, worker):
        """Test successful task retry."""
        worker.config.queue.retry_delay = 1
        worker.redis_pool = MagicMock()
        
        mock_redis = AsyncMock()
        with patch('redis.asyncio.Redis', return_value=mock_redis):
            task = {'retry_count': 1}
            
            with patch('asyncio.sleep') as mock_sleep:
                await worker._retry_task(task, 'test-worker')
                
                assert task['retry_count'] == 2
                mock_sleep.assert_called_once_with(2)  # retry_delay * 2^(retry_count-1)
                mock_redis.lpush.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_task_first_retry(self, worker):
        """Test first retry attempt."""
        worker.config.queue.retry_delay = 5
        worker.redis_pool = MagicMock()
        
        mock_redis = AsyncMock()
        with patch('redis.asyncio.Redis', return_value=mock_redis):
            task = {}  # No retry_count yet
            
            with patch('asyncio.sleep') as mock_sleep:
                await worker._retry_task(task, 'test-worker')
                
                assert task['retry_count'] == 1
                mock_sleep.assert_called_once_with(5)  # retry_delay * 2^0

    @pytest.mark.asyncio
    async def test_retry_task_redis_error(self, worker):
        """Test retry task with Redis error."""
        worker.config.queue.retry_delay = 1
        worker.redis_pool = MagicMock()
        
        mock_redis = AsyncMock()
        mock_redis.lpush.side_effect = Exception("Redis error")
        
        with patch('redis.asyncio.Redis', return_value=mock_redis):
            task = {'retry_count': 0}
            
            with patch('asyncio.sleep'):
                # Should not raise exception, just log error
                await worker._retry_task(task, 'test-worker')
                
                assert task['retry_count'] == 1

    @pytest.mark.asyncio
    async def test_start_full_success_flow(self, worker):
        """Test complete successful start flow including all setup steps."""
        with patch.object(worker, '_setup_redis') as mock_redis:
            with patch.object(worker, '_setup_database_managers') as mock_db:
                with patch.object(worker, '_setup_embedding_clients') as mock_embed:
                    with patch.object(worker, '_setup_preprocessing_pipelines') as mock_preprocess:
                        with patch.object(worker, '_run_workers') as mock_run:
                            await worker.start()
                            
                            # Verify all setup methods were called in order
                            mock_redis.assert_called_once()
                            mock_db.assert_called_once()
                            mock_embed.assert_called_once()
                            mock_preprocess.assert_called_once()
                            mock_run.assert_called_once()
                            
                            assert worker.running is True

    @pytest.mark.asyncio
    async def test_start_with_embedding_setup_failure(self, worker):
        """Test start method with embedding client setup failure."""
        with patch.object(worker, '_setup_redis'):
            with patch.object(worker, '_setup_database_managers'):
                with patch.object(worker, '_setup_embedding_clients', side_effect=EmbeddingError("Client failed")):
                    with patch.object(worker, 'stop') as mock_stop:
                        with pytest.raises(WorkerError):
                            await worker.start()
                        
                        mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_with_database_setup_failure(self, worker):
        """Test start method with database setup failure."""
        with patch.object(worker, '_setup_redis'):
            with patch.object(worker, '_setup_database_managers', side_effect=DatabaseError("DB failed")):
                with patch.object(worker, 'stop') as mock_stop:
                    with pytest.raises(WorkerError):
                        await worker.start()
                    
                    mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_with_client_close_errors(self, worker):
        """Test stop method with client close errors."""
        # Setup mock clients with close methods that raise exceptions
        mock_client1 = MagicMock()
        mock_client1.close = AsyncMock(side_effect=Exception("Close failed"))
        worker.embedding_clients['client1'] = mock_client1
        
        mock_client2 = MagicMock()  # No close method
        worker.embedding_clients['client2'] = mock_client2
        
        # Setup mock database manager with close error
        mock_manager = MagicMock()
        mock_manager.close = AsyncMock(side_effect=Exception("DB close failed"))
        worker.database_managers['db1'] = mock_manager
        
        # Setup mock Redis pool with disconnect error
        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock(side_effect=Exception("Redis close failed"))
        worker.redis_pool = mock_pool
        
        # Should not raise exception despite all the errors
        await worker.stop()
        
        assert worker.running is False

    @pytest.mark.asyncio
    async def test_run_workers_keyboard_interrupt(self, worker):
        """Test _run_workers with KeyboardInterrupt handling."""
        worker.config.workers.concurrency = 2
        
        async def mock_worker_loop(worker_name):
            # Simulate a running worker that gets interrupted
            await asyncio.sleep(0.1)
            raise KeyboardInterrupt()
        
        with patch.object(worker, '_worker_loop', side_effect=mock_worker_loop):
            with patch('asyncio.create_task') as mock_create_task:
                # Create mock tasks that can be cancelled
                mock_task1 = MagicMock()
                mock_task2 = MagicMock()
                mock_create_task.side_effect = [mock_task1, mock_task2]
                
                with patch('asyncio.gather', side_effect=KeyboardInterrupt()) as mock_gather:
                    await worker._run_workers()
                    
                    # Verify tasks were created and cancelled
                    mock_task1.cancel.assert_called_once()
                    mock_task2.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_worker_loop_with_task_processing_error(self, worker):
        """Test worker loop with task processing error and recovery."""
        worker.running = True
        call_count = 0
        
        async def mock_get_task():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return {'event': {}, 'table_config': {}}  # Valid task
            elif call_count == 2:
                return {'event': {}, 'table_config': {}}  # Another task
            else:
                worker.running = False  # Stop after 2 iterations
                return None
        
        async def mock_process_task(task, worker_name):
            if 'event' in task:
                raise Exception("Processing failed")
        
        with patch.object(worker, '_get_task', side_effect=mock_get_task):
            with patch.object(worker, '_process_task', side_effect=mock_process_task):
                with patch('asyncio.sleep') as mock_sleep:
                    await worker._worker_loop('test-worker')
                    
                    # Should have called sleep for error recovery
                    assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    async def test_worker_loop_no_tasks_available(self, worker):
        """Test worker loop when no tasks are available."""
        worker.running = True
        call_count = 0
        
        async def mock_get_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # No tasks available
            else:
                worker.running = False  # Stop after a few iterations
                return None
        
        with patch.object(worker, '_get_task', side_effect=mock_get_task):
            with patch('asyncio.sleep') as mock_sleep:
                await worker._worker_loop('test-worker')
                
                # Should have called sleep for brief pause when no tasks
                assert mock_sleep.call_count >= 2

    @pytest.mark.asyncio
    async def test_preprocess_text_pipeline_fallback(self, worker):
        """Test preprocessing with pipeline fallback to basic cleanup."""
        # Setup basic cleanup pipeline
        mock_basic_pipeline = AsyncMock()
        mock_basic_pipeline.process.return_value = "basic processed text"
        worker.preprocessing_pipelines['basic_cleanup'] = mock_basic_pipeline
        
        table_config = {
            'preprocessing': {
                'enabled': True,
                'cleaning_config': 'missing_config'
            }
        }
        
        result = await worker._preprocess_text('test text', table_config, 'test-worker')
        
        assert result == "basic processed text"
        mock_basic_pipeline.process.assert_called_once_with('test text')

    @pytest.mark.asyncio
    async def test_preprocess_text_no_pipeline_available(self, worker):
        """Test preprocessing when no pipeline is available at all."""
        # No pipelines in worker.preprocessing_pipelines
        
        table_config = {
            'preprocessing': {
                'enabled': True,
                'cleaning_config': 'missing_config'
            }
        }
        
        result = await worker._preprocess_text('test text', table_config, 'test-worker')
        
        # Should return original text when no pipeline available
        assert result == "test text"

    @pytest.mark.asyncio
    async def test_preprocess_text_disabled(self, worker):
        """Test text preprocessing when disabled in config."""
        table_config = {
            'preprocessing': {
                'enabled': False,
                'cleaning_config': 'some_config'
            }
        }
        
        result = await worker._preprocess_text('test text', table_config, 'test-worker')
        
        # Should return original text when preprocessing disabled
        assert result == "test text"

    @pytest.mark.asyncio
    async def test_preprocess_text_no_config(self, worker):
        """Test text preprocessing when no preprocessing config exists."""
        table_config = {}  # No preprocessing config
        
        result = await worker._preprocess_text('test text', table_config, 'test-worker')
        
        # Should return original text when no config
        assert result == "test text"

    @pytest.mark.asyncio
    async def test_preprocess_text_processing_exception(self, worker):
        """Test text preprocessing when pipeline processing raises exception."""
        mock_pipeline = AsyncMock()
        mock_pipeline.process.side_effect = Exception("Processing failed")
        worker.preprocessing_pipelines['test_config'] = mock_pipeline
        
        table_config = {
            'preprocessing': {
                'enabled': True,
                'cleaning_config': 'test_config'
            }
        }
        
        result = await worker._preprocess_text('test text', table_config, 'test-worker')
        
        # Should return original text when processing fails
        assert result == "test text"

    @pytest.mark.asyncio
    async def test_process_task_full_flow(self, worker):
        """Test complete task processing flow."""
        task = {
            'event': {
                'id': {
                    'id': 123,
                    'content': 'Test content to vectorize'
                }
            },
            'table_config': {
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'id_column': 'id',
                'vector_column': 'content_vector',
                'model_column': 'content_model',
                'database_name': 'test_db',
                'embedding_config': 'test_embedding',
                'preprocessing': {
                    'enabled': True,
                    'cleaning_config': 'basic_cleanup'
                }
            }
        }
        
        # Mock preprocessing pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.process.return_value = "Processed content"
        worker.preprocessing_pipelines['basic_cleanup'] = mock_pipeline
        
        # Mock embedding client
        mock_client = AsyncMock()
        mock_client.embed_texts.return_value = [[0.1, 0.2, 0.3]]
        worker.embedding_clients['test_embedding'] = mock_client
        
        # Mock database manager
        mock_manager = MagicMock()
        mock_conn = AsyncMock()
        mock_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        worker.database_managers['test_db'] = mock_manager
        
        # Mock embedding config
        mock_embedding_config = MagicMock()
        mock_embedding_config.model = "test-model"
        worker.config.embeddings = {'test_embedding': mock_embedding_config}
        
        await worker._process_task(task, "worker1")
        
        # Verify the full flow
        mock_pipeline.process.assert_called_once_with("Test content to vectorize")
        mock_client.embed_texts.assert_called_once_with(["Processed content"])
        mock_conn.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_task_no_text(self, worker):
        """Test task processing with no extractable text."""
        task = {
            'event': {
                'id': {
                    'id': 123
                    # Missing 'content' field
                }
            },
            'table_config': {
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'id_column': 'id',
                'vector_column': 'content_vector',
                'model_column': 'content_model',
                'database_name': 'test_db',
                'embedding_config': 'test_embedding'
            }
        }
        
        # Should not process anything if no text
        await worker._process_task(task, "worker1")
        
        # No embedding client should be called
        assert 'test_embedding' not in worker.embedding_clients or not hasattr(worker.embedding_clients.get('test_embedding'), 'embed_texts')

    @pytest.mark.asyncio
    async def test_process_task_empty_processed_text(self, worker):
        """Test task processing with empty text after preprocessing."""
        task = {
            'event': {
                'id': {
                    'id': 123,
                    'content': 'Original content'
                }
            },
            'table_config': {
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'id_column': 'id',
                'vector_column': 'content_vector',
                'model_column': 'content_model',
                'database_name': 'test_db',
                'embedding_config': 'test_embedding',
                'preprocessing': {
                    'enabled': True,
                    'cleaning_config': 'basic_cleanup'
                }
            }
        }
        
        # Mock preprocessing pipeline that returns empty string
        mock_pipeline = AsyncMock()
        mock_pipeline.process.return_value = ""
        worker.preprocessing_pipelines['basic_cleanup'] = mock_pipeline
        
        await worker._process_task(task, "worker1")
        
        # Should process with preprocessing but not embed empty text
        mock_pipeline.process.assert_called_once_with("Original content")

    @pytest.mark.asyncio
    async def test_process_task_missing_embedding_client(self, worker):
        """Test task processing with missing embedding client."""
        task = {
            'event': {
                'id': {
                    'id': 123,
                    'content': 'Test content'
                }
            },
            'table_config': {
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'id_column': 'id',
                'vector_column': 'content_vector',
                'model_column': 'content_model',
                'database_name': 'test_db',
                'embedding_config': 'missing_embedding'
            }
        }
        
        # No embedding client for 'missing_embedding'
        
        # Should not raise exception, just log error and return
        await worker._process_task(task, "worker1")

    @pytest.mark.asyncio
    async def test_process_task_exception_with_retry(self, worker):
        """Test task processing exception that triggers retry."""
        task = {
            'event': {
                'id': {
                    'id': 123,
                    'content': 'Test content'
                }
            },
            'table_config': {
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'id_column': 'id',
                'vector_column': 'content_vector',
                'model_column': 'content_model',
                'database_name': 'test_db',
                'embedding_config': 'test_embedding'
            },
            'retry_count': 0
        }
        
        # Mock embedding client that raises exception
        mock_client = AsyncMock()
        mock_client.embed_texts.side_effect = Exception("Embedding failed")
        worker.embedding_clients['test_embedding'] = mock_client
        
        # Mock config
        worker.config.queue.max_retries = 3
        
        # Mock _retry_task method
        worker._retry_task = AsyncMock()
        
        await worker._process_task(task, "worker1")
        
        # Should call retry
        worker._retry_task.assert_called_once_with(task, "worker1")

    @pytest.mark.asyncio
    async def test_process_task_exception_max_retries_exceeded(self, worker):
        """Test process_task with exception and max retries exceeded."""
        # Setup mocks
        worker.redis_pool = MagicMock()
        worker.embedding_clients = {'test_embedding': MagicMock()}
        worker.database_managers = {'test_db': MagicMock()}
        
        # Mock task
        task = {
            'event': {'id': 1, 'content': 'test content'},
            'table_config': {
                'database': 'test_db',
                'schema': 'public',
                'table': 'test_table',
                'text_column': 'content',
                'embedding_config': 'test_embedding',
                'vector_column': 'content_vector',
                'model_column': 'content_model'
            },
            'retry_count': 4  # Exceeds max retries (3)
        }
        
        # Mock _extract_text_from_event to raise exception
        worker._extract_text_from_event = MagicMock(side_effect=Exception("Extraction failed"))
        
        # Should not retry when max retries exceeded
        worker._retry_task = AsyncMock()
        
        await worker._process_task(task, "test_worker")
        
        # Should not call retry since max retries exceeded
        worker._retry_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_redis_setup_failure(self, worker):
        """Test start() method with Redis setup failure."""
        with patch.object(worker, '_setup_redis', new_callable=AsyncMock) as mock_setup_redis:
            with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                mock_setup_redis.side_effect = WorkerError("Redis connection failed")
                
                with pytest.raises(WorkerError) as exc_info:
                    await worker.start()
                
                assert "Worker startup failed" in str(exc_info.value)
                mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_database_setup_failure(self, worker):
        """Test start() method with database setup failure."""
        with patch.object(worker, '_setup_redis', new_callable=AsyncMock):
            with patch.object(worker, '_setup_database_managers', new_callable=AsyncMock) as mock_setup_db:
                with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                    mock_setup_db.side_effect = DatabaseError("Database setup failed")
                    
                    with pytest.raises(WorkerError) as exc_info:
                        await worker.start()
                    
                    assert "Worker startup failed" in str(exc_info.value)
                    mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_embedding_setup_failure(self, worker):
        """Test start() method with embedding setup failure."""
        with patch.object(worker, '_setup_redis', new_callable=AsyncMock):
            with patch.object(worker, '_setup_database_managers', new_callable=AsyncMock):
                with patch.object(worker, '_setup_embedding_clients', new_callable=AsyncMock) as mock_setup_emb:
                    with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                        mock_setup_emb.side_effect = EmbeddingError("Embedding setup failed")
                        
                        with pytest.raises(WorkerError) as exc_info:
                            await worker.start()
                        
                        assert "Worker startup failed" in str(exc_info.value)
                        mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_preprocessing_setup_failure(self, worker):
        """Test start() method with preprocessing setup failure."""
        with patch.object(worker, '_setup_redis', new_callable=AsyncMock):
            with patch.object(worker, '_setup_database_managers', new_callable=AsyncMock):
                with patch.object(worker, '_setup_embedding_clients', new_callable=AsyncMock):
                    with patch.object(worker, '_setup_preprocessing_pipelines', new_callable=AsyncMock) as mock_setup_prep:
                        with patch.object(worker, 'stop', new_callable=AsyncMock) as mock_stop:
                            mock_setup_prep.side_effect = Exception("Preprocessing setup failed")
                            
                            with pytest.raises(WorkerError) as exc_info:
                                await worker.start()
                            
                            assert "Worker startup failed" in str(exc_info.value)
                            mock_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_embedding_client_close_error(self, worker):
        """Test stop() method with embedding client close error."""
        # Setup mock embedding client with close method that raises exception
        mock_client = MagicMock()
        mock_client.close = AsyncMock(side_effect=Exception("Close failed"))
        worker.embedding_clients = {'test_client': mock_client}
        
        # Should not raise exception, just log warning
        await worker.stop()
        
        mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_database_manager_close_error(self, worker):
        """Test stop() method with database manager close error."""
        # Setup mock database manager with close method that raises exception
        mock_manager = MagicMock()
        mock_manager.close = AsyncMock(side_effect=Exception("Close failed"))
        worker.database_managers = {'test_db': mock_manager}
        
        # Should not raise exception, just log warning
        await worker.stop()
        
        mock_manager.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_redis_disconnect_error(self, worker):
        """Test stop() method with Redis disconnect error."""
        # Setup mock Redis pool with disconnect method that raises exception
        mock_pool = MagicMock()
        mock_pool.disconnect = AsyncMock(side_effect=Exception("Disconnect failed"))
        worker.redis_pool = mock_pool
        
        # Should not raise exception, just log warning
        await worker.stop()
        
        mock_pool.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_task_redis_error(self, worker):
        """Test _get_task with Redis error."""
        with patch('redis.asyncio.Redis') as mock_redis_class:
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            mock_redis.brpop.side_effect = Exception("Redis error")
            
            worker.redis_pool = MagicMock()
            
            result = await worker._get_task()
            
            assert result is None

    @pytest.mark.asyncio
    async def test_setup_preprocessing_pipelines_error_handling(self, worker):
        """Test preprocessing pipeline setup with error handling."""
        # Add a cleaning config that will cause an error
        mock_cleaning_config = MagicMock()
        worker.config.cleaning_configs['failing_config'] = mock_cleaning_config
        
        with patch('v10r.worker.PreprocessingFactory') as mock_factory_class:
            mock_factory = MagicMock()
            mock_factory_class.return_value = mock_factory
            
            # Make create_pipeline fail for the first config
            mock_factory.create_pipeline.side_effect = Exception("Pipeline creation failed")
            
            mock_basic_pipeline = MagicMock()
            mock_factory.get_basic_pipeline.return_value = mock_basic_pipeline
            
            # Should not raise exception, should create basic pipeline
            await worker._setup_preprocessing_pipelines()
            
            assert "basic_cleanup" in worker.preprocessing_pipelines
            mock_factory.get_basic_pipeline.assert_called_once() 