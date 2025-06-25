"""
Redis queue worker for v10r vectorizer.

This module implements the worker processes that consume vectorization tasks
from Redis queues, generate embeddings, and update database vectors.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any

import redis.asyncio as redis
import asyncpg

from .config import V10rConfig
from .database.connection import DatabaseManager
from .embedding.factory import EmbeddingClientFactory
from .preprocessing.factory import PreprocessingFactory
from .exceptions import WorkerError, EmbeddingError, DatabaseError

logger = logging.getLogger(__name__)


class VectorizerWorker:
    """
    Worker process for handling vectorization tasks.
    
    Consumes tasks from Redis queues, generates embeddings, and updates
    database vectors with proper error handling and retry logic.
    """

    def __init__(self, config: V10rConfig):
        self.config = config
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.database_managers: Dict[str, DatabaseManager] = {}
        self.embedding_clients: Dict[str, Any] = {}
        self.preprocessing_pipelines: Dict[str, Any] = {}
        self.running = False
        self.worker_id = f"worker-{int(time.time())}"

    async def start(self) -> None:
        """Start the worker service."""
        logger.info(f"Starting v10r worker service (ID: {self.worker_id})")
        
        try:
            # Initialize Redis connection
            await self._setup_redis()
            
            # Initialize database managers
            await self._setup_database_managers()
            
            # Initialize embedding clients
            await self._setup_embedding_clients()
            
            # Initialize preprocessing pipelines
            await self._setup_preprocessing_pipelines()
            
            self.running = True
            logger.info(f"Worker {self.worker_id} started successfully")
            
            # Start worker processes
            await self._run_workers()
            
        except Exception as e:
            logger.error(f"Failed to start worker service: {e}")
            await self.stop()
            raise WorkerError(f"Worker startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the worker service and cleanup resources."""
        logger.info(f"Stopping worker {self.worker_id}...")
        self.running = False
        
        # Close embedding clients
        for config_name, client in self.embedding_clients.items():
            try:
                if hasattr(client, 'close'):
                    await client.close()
                logger.debug(f"Closed embedding client: {config_name}")
            except Exception as e:
                logger.warning(f"Error closing embedding client {config_name}: {e}")
        
        # Close database managers
        for db_name, manager in self.database_managers.items():
            try:
                await manager.close()
                logger.debug(f"Closed database manager: {db_name}")
            except Exception as e:
                logger.warning(f"Error closing database manager {db_name}: {e}")
        
        # Close Redis connection
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect()
                logger.debug("Closed Redis connection pool")
            except Exception as e:
                logger.warning(f"Error closing Redis pool: {e}")
        
        logger.info(f"Worker {self.worker_id} stopped")

    async def _setup_redis(self) -> None:
        """Initialize Redis connection pool."""
        try:
            redis_config = self.config.queue.connection
            
            self.redis_pool = redis.ConnectionPool(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                db=redis_config.get('db', 0),
                decode_responses=False  # We'll handle JSON encoding/decoding manually
            )
            
            # Test connection
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            await redis_client.ping()
            
            logger.info(f"Connected to Redis at {redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}")
            
        except Exception as e:
            raise WorkerError(f"Failed to connect to Redis: {e}") from e

    async def _setup_database_managers(self) -> None:
        """Initialize database managers for all configured databases."""
        for db_config in self.config.databases:
            try:
                manager = DatabaseManager(db_config.connection)
                self.database_managers[db_config.name] = manager
                logger.debug(f"Initialized database manager for: {db_config.name}")
            except Exception as e:
                logger.error(f"Failed to setup database manager for {db_config.name}: {e}")
                raise DatabaseError(f"Database manager setup failed: {e}") from e

    async def _setup_embedding_clients(self) -> None:
        """Initialize embedding clients for all configured providers."""
        factory = EmbeddingClientFactory()
        
        for config_name, embedding_config in self.config.embeddings.items():
            try:
                client = await factory.create_client(embedding_config)
                self.embedding_clients[config_name] = client
                logger.debug(f"Initialized embedding client: {config_name} ({embedding_config.provider})")
            except Exception as e:
                logger.error(f"Failed to setup embedding client {config_name}: {e}")
                raise EmbeddingError(f"Embedding client setup failed: {e}") from e

    async def _setup_preprocessing_pipelines(self) -> None:
        """Initialize preprocessing pipelines for all configured cleaners."""
        factory = PreprocessingFactory()
        
        # Create pipelines for all cleaning configs
        for config_name, cleaning_config in self.config.cleaning_configs.items():
            try:
                pipeline = factory.create_pipeline(cleaning_config)
                self.preprocessing_pipelines[config_name] = pipeline
                logger.debug(f"Initialized preprocessing pipeline: {config_name}")
            except Exception as e:
                logger.error(f"Failed to setup preprocessing pipeline {config_name}: {e}")
                # Don't raise - preprocessing is optional
        
        # Always have a basic pipeline available
        if "basic_cleanup" not in self.preprocessing_pipelines:
            basic_pipeline = factory.get_basic_pipeline()
            self.preprocessing_pipelines["basic_cleanup"] = basic_pipeline
            logger.debug("Created default basic preprocessing pipeline")

    async def _run_workers(self) -> None:
        """Run worker processes with configured concurrency."""
        concurrency = self.config.workers.concurrency
        logger.info(f"Starting {concurrency} worker processes")
        
        # Create worker tasks
        worker_tasks = []
        for i in range(concurrency):
            task = asyncio.create_task(self._worker_loop(f"{self.worker_id}-{i}"))
            worker_tasks.append(task)
        
        try:
            # Wait for all workers to complete
            await asyncio.gather(*worker_tasks)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down workers...")
            for task in worker_tasks:
                task.cancel()
            await asyncio.gather(*worker_tasks, return_exceptions=True)

    async def _worker_loop(self, worker_name: str) -> None:
        """Main worker loop for processing tasks."""
        logger.info(f"Started worker: {worker_name}")
        
        while self.running:
            try:
                # Get task from queue with timeout
                task = await self._get_task()
                
                if task:
                    await self._process_task(task, worker_name)
                else:
                    # No task available, brief sleep to avoid busy waiting
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in worker {worker_name}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
        
        logger.info(f"Worker {worker_name} shutting down")

    async def _get_task(self) -> Optional[Dict]:
        """Get next task from Redis queue."""
        try:
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            queue_name = self.config.queue.name
            
            # Blocking pop with timeout
            task_data = await redis_client.brpop(queue_name, timeout=1)
            
            if task_data:
                task_json = task_data[1].decode('utf-8')
                return json.loads(task_json)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get task from queue: {e}")
            return None

    async def _process_task(self, task: Dict, worker_name: str) -> None:
        """Process a single vectorization task."""
        try:
            event = task['event']
            table_config = task['table_config']
            
            logger.debug(f"{worker_name}: Processing task for {table_config['schema']}.{table_config['table']}")
            
            # Get text from the event data
            text = self._extract_text_from_event(event, table_config)
            
            if not text or not text.strip():
                logger.debug(f"{worker_name}: No text to vectorize, skipping")
                return
            
            # Apply preprocessing if configured
            processed_text = await self._preprocess_text(text, table_config, worker_name)
            
            if not processed_text or not processed_text.strip():
                logger.debug(f"{worker_name}: No text after preprocessing, skipping")
                return
            
            # Get embedding client
            embedding_config_name = table_config['embedding_config']
            client = self.embedding_clients.get(embedding_config_name)
            
            if not client:
                logger.error(f"No embedding client found for config: {embedding_config_name}")
                return
            
            # Generate embedding
            embeddings = await client.embed_texts([processed_text])
            embedding = embeddings[0]
            
            # Update database with vector
            await self._update_vector_in_database(
                event, table_config, embedding, worker_name
            )
            
            logger.debug(f"{worker_name}: Successfully processed task for {table_config['schema']}.{table_config['table']}")
            
        except Exception as e:
            logger.error(f"{worker_name}: Failed to process task: {e}")
            
            # Check if task should be retried
            retry_count = task.get('retry_count', 0)
            max_retries = self.config.queue.max_retries
            
            if retry_count < max_retries:
                await self._retry_task(task, worker_name)
            else:
                logger.error(f"{worker_name}: Task exceeded max retries ({max_retries}), dropping")

    def _extract_text_from_event(self, event: Dict, table_config: Dict) -> Optional[str]:
        """Extract text content from the event data."""
        try:
            row_data = event.get('id', {})
            text_column = table_config['text_column']
            
            text = row_data.get(text_column)
            return text if text else None
            
        except Exception as e:
            logger.error(f"Failed to extract text from event: {e}")
            return None

    async def _preprocess_text(self, text: str, table_config: Dict, worker_name: str) -> str:
        """Apply preprocessing to text based on table configuration."""
        try:
            # Check if preprocessing is enabled for this table
            preprocessing_config = table_config.get('preprocessing')
            
            if not preprocessing_config or not preprocessing_config.get('enabled', False):
                # No preprocessing configured, return original text
                return text
            
            # Get the cleaning configuration name
            cleaning_config_name = preprocessing_config.get('cleaning_config', 'basic_cleanup')
            
            # Get the preprocessing pipeline
            pipeline = self.preprocessing_pipelines.get(cleaning_config_name)
            
            if not pipeline:
                logger.warning(f"{worker_name}: No preprocessing pipeline found for '{cleaning_config_name}', using basic cleanup")
                pipeline = self.preprocessing_pipelines.get('basic_cleanup')
            
            if not pipeline:
                logger.error(f"{worker_name}: No preprocessing pipeline available, returning original text")
                return text
            
            # Apply preprocessing
            processed_text = await pipeline.process(text)
            
            logger.debug(f"{worker_name}: Preprocessed text (original: {len(text)} chars, processed: {len(processed_text)} chars)")
            
            return processed_text
            
        except Exception as e:
            logger.error(f"{worker_name}: Text preprocessing failed: {e}")
            # Return original text if preprocessing fails
            return text

    async def _update_vector_in_database(
        self, 
        event: Dict, 
        table_config: Dict, 
        embedding: List[float],
        worker_name: str
    ) -> None:
        """Update the vector column in the database."""
        try:
            # Get database manager
            db_name = table_config['database_name']
            manager = self.database_managers.get(db_name)
            
            if not manager:
                logger.error(f"No database manager found for: {db_name}")
                return
            
            # Get row ID
            row_data = event.get('id', {})
            id_column = table_config['id_column']
            row_id = row_data.get(id_column)
            
            if row_id is None:
                logger.error(f"No ID found in event data for column: {id_column}")
                return
            
            # Build update SQL
            schema = table_config['schema']
            table = table_config['table']
            vector_column = table_config['vector_column']
            model_column = table_config['model_column']
            
            # Get embedding config for model name
            embedding_config_name = table_config['embedding_config']
            embedding_config = self.config.embeddings.get(embedding_config_name)
            model_name = embedding_config.model if embedding_config else 'unknown'
            
            update_sql = f'''
                UPDATE {schema}.{table}
                SET {vector_column} = $1,
                    {model_column} = $2,
                    {vector_column.replace("_vector", "_vector_created_at")} = NOW()
                WHERE {id_column} = $3
            '''
            
            # Execute update
            async with manager.pool.acquire() as conn:
                await conn.execute(update_sql, embedding, model_name, row_id)
            
            logger.debug(f"{worker_name}: Updated vector for {schema}.{table} ID {row_id}")
            
        except Exception as e:
            logger.error(f"{worker_name}: Failed to update vector in database: {e}")
            raise

    async def _retry_task(self, task: Dict, worker_name: str) -> None:
        """Retry a failed task with exponential backoff."""
        try:
            task['retry_count'] = task.get('retry_count', 0) + 1
            retry_delay = self.config.queue.retry_delay * (2 ** (task['retry_count'] - 1))
            
            logger.warning(f"{worker_name}: Retrying task in {retry_delay}s (attempt {task['retry_count']})")
            
            # Wait before retrying
            await asyncio.sleep(retry_delay)
            
            # Re-queue the task
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            task_json = json.dumps(task)
            await redis_client.lpush(self.config.queue.name, task_json)
            
        except Exception as e:
            logger.error(f"{worker_name}: Failed to retry task: {e}") 