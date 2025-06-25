"""
PostgreSQL NOTIFY/LISTEN service for v10r vectorizer.

This module implements the core event listener that receives PostgreSQL 
NOTIFY events and queues vectorization tasks for processing.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Set

import redis.asyncio as redis
import asyncpg

from .config import V10rConfig
from .database.connection import DatabaseManager
from .exceptions import ListenerError, DatabaseConnectionError

logger = logging.getLogger(__name__)


class VectorListener:
    """
    PostgreSQL NOTIFY/LISTEN service for vectorization events.
    
    Listens for database NOTIFY events and queues vectorization tasks
    in Redis for worker processes to handle.
    """

    def __init__(self, config: V10rConfig, channel: str = "v10r_events"):
        self.config = config
        self.channel = channel
        self.database_managers: Dict[str, DatabaseManager] = {}
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.listeners: Dict[str, asyncpg.Connection] = {}
        self.running = False

    async def start(self) -> None:
        """Start the listener service."""
        logger.info(f"Starting v10r listener service on channel '{self.channel}'")
        
        try:
            # Initialize Redis connection
            await self._setup_redis()
            
            # Initialize database connections and listeners
            await self._setup_database_listeners()
            
            self.running = True
            logger.info("v10r listener service started successfully")
            
            # Keep the service running
            await self._run_forever()
            
        except Exception as e:
            logger.error(f"Failed to start listener service: {e}")
            await self.stop()
            raise ListenerError(f"Listener startup failed: {e}") from e

    async def stop(self) -> None:
        """Stop the listener service and cleanup resources."""
        logger.info("Stopping v10r listener service...")
        self.running = False
        
        # Close database listeners
        for db_name, conn in self.listeners.items():
            try:
                if not conn.is_closed():
                    await conn.close()
                logger.debug(f"Closed listener connection for database: {db_name}")
            except Exception as e:
                logger.warning(f"Error closing listener for {db_name}: {e}")
        
        # Close database managers
        for db_name, manager in self.database_managers.items():
            try:
                await manager.close()
                logger.debug(f"Closed database manager for: {db_name}")
            except Exception as e:
                logger.warning(f"Error closing database manager {db_name}: {e}")
        
        # Close Redis connection
        if self.redis_pool:
            try:
                await self.redis_pool.disconnect()
                logger.debug("Closed Redis connection pool")
            except Exception as e:
                logger.warning(f"Error closing Redis pool: {e}")
        
        logger.info("v10r listener service stopped")

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
            raise ListenerError(f"Failed to connect to Redis: {e}") from e

    async def _setup_database_listeners(self) -> None:
        """Initialize database connections and NOTIFY listeners."""
        for db_config in self.config.databases:
            try:
                # Create database manager
                manager = DatabaseManager(db_config.connection)
                self.database_managers[db_config.name] = manager
                
                # Create dedicated listener connection
                conn = await asyncpg.connect(
                    host=db_config.connection.host,
                    port=db_config.connection.port,
                    database=db_config.connection.database,
                    user=db_config.connection.user,
                    password=db_config.connection.password,
                )
                
                # Start listening on the channel
                await conn.add_listener(self.channel, self._handle_notification)
                self.listeners[db_config.name] = conn
                
                logger.info(f"Started listener for database: {db_config.name}")
                
            except Exception as e:
                logger.error(f"Failed to setup listener for database {db_config.name}: {e}")
                raise DatabaseConnectionError(f"Database listener setup failed: {e}") from e

    async def _handle_notification(self, connection: asyncpg.Connection, pid: int, channel: str, payload: str) -> None:
        """
        Handle incoming PostgreSQL NOTIFY events.
        
        Parses the notification payload and queues vectorization tasks.
        """
        try:
            # Parse the notification payload
            event_data = json.loads(payload)
            
            logger.debug(f"Received notification: {event_data}")
            
            # Validate required fields
            required_fields = ['database', 'schema', 'table', 'operation', 'id']
            for field in required_fields:
                if field not in event_data:
                    logger.warning(f"Missing required field '{field}' in notification: {payload}")
                    return
            
            # Find matching table configuration
            table_config = self._find_table_config(
                event_data['database'],
                event_data['schema'], 
                event_data['table']
            )
            
            if not table_config:
                logger.debug(f"No configuration found for table {event_data['schema']}.{event_data['table']}")
                return
            
            # Create vectorization task
            task = {
                'event': event_data,
                'table_config': table_config,
                'timestamp': asyncio.get_event_loop().time(),
                'retry_count': 0
            }
            
            # Queue the task
            await self._queue_task(task)
            
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in notification payload: {payload}, error: {e}")
        except Exception as e:
            logger.error(f"Error handling notification: {e}")

    def _find_table_config(self, database: str, schema: str, table: str) -> Optional[Dict]:
        """Find table configuration for the given database/schema/table."""
        for db_config in self.config.databases:
            if db_config.name == database or db_config.connection.database == database:
                for table_config in db_config.tables:
                    if table_config.schema == schema and table_config.table == table:
                        return {
                            'schema': table_config.schema,
                            'table': table_config.table,
                            'text_column': table_config.text_column,
                            'id_column': table_config.id_column,
                            'vector_column': getattr(table_config, 'vector_column', f"{table_config.text_column}_vector"),
                            'model_column': getattr(table_config, 'model_column', f"{table_config.text_column}_embedding_model"),
                            'embedding_config': table_config.embedding_config,
                            'database_name': db_config.name
                        }
        return None

    async def _queue_task(self, task: Dict) -> None:
        """Queue vectorization task in Redis."""
        try:
            redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Serialize task
            task_json = json.dumps(task)
            
            # Push to the main task queue 
            queue_name = self.config.queue.name
            await redis_client.lpush(queue_name, task_json)
            
            logger.debug(f"Queued task for {task['table_config']['schema']}.{task['table_config']['table']}")
            
        except Exception as e:
            logger.error(f"Failed to queue task: {e}")
            # In a production system, you might want to implement a dead letter queue here

    async def _run_forever(self) -> None:
        """Keep the service running and handle reconnections."""
        while self.running:
            try:
                # Check connection health
                for db_name, conn in self.listeners.items():
                    if conn.is_closed():
                        logger.warning(f"Database connection {db_name} is closed, attempting reconnect...")
                        await self._reconnect_database(db_name)
                
                # Sleep briefly to avoid busy waiting
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def _reconnect_database(self, db_name: str) -> None:
        """Reconnect to a specific database."""
        try:
            # Find database config
            db_config = None
            for config in self.config.databases:
                if config.name == db_name:
                    db_config = config
                    break
            
            if not db_config:
                logger.error(f"No configuration found for database: {db_name}")
                return
            
            # Close existing connection
            if db_name in self.listeners:
                try:
                    await self.listeners[db_name].close()
                except:
                    pass
            
            # Create new connection
            conn = await asyncpg.connect(
                host=db_config.connection.host,
                port=db_config.connection.port,
                database=db_config.connection.database,
                user=db_config.connection.user,
                password=db_config.connection.password,
            )
            
            # Start listening
            await conn.add_listener(self.channel, self._handle_notification)
            self.listeners[db_name] = conn
            
            logger.info(f"Reconnected to database: {db_name}")
            
        except Exception as e:
            logger.error(f"Failed to reconnect to database {db_name}: {e}") 