"""
Table registration service for v10r vectorizer.

This module implements the table registration logic that sets up vectorization
for database tables, including schema modifications and trigger installation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import V10rConfig
from .database.connection import DatabaseManager
from .database.introspection import SchemaIntrospector
from .schema.operations import SafeSchemaOperations, OperationMode
from .schema.collision_detector import ColumnCollisionDetector, CollisionSeverity
from .exceptions import RegistrationError, SchemaError, ValidationError

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """Result of table registration operation."""
    
    success: bool
    vector_column: Optional[str] = None
    model_column: Optional[str] = None
    timestamp_column: Optional[str] = None
    warnings: List[str] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TableRegistrationService:
    """
    Service for registering tables for vectorization.
    
    Handles schema modifications, collision detection, and trigger setup
    required to enable automatic vectorization for a table.
    """

    def __init__(self, config: V10rConfig):
        self.config = config
        self.database_managers: Dict[str, DatabaseManager] = {}

    async def register_table(
        self,
        database: str,
        schema: str,
        table: str,
        text_column: str,
        embedding_config: str,
        vector_column: Optional[str] = None,
        model_column: Optional[str] = None,
        collision_strategy: str = "interactive",
    ) -> RegistrationResult:
        """
        Register a table for vectorization.
        
        Args:
            database: Database name from config
            schema: Database schema name
            table: Table name
            text_column: Column containing text to vectorize
            embedding_config: Embedding configuration name
            vector_column: Vector column name (auto-generated if None)
            model_column: Model tracking column name (auto-generated if None)
            collision_strategy: How to handle column name collisions
            
        Returns:
            RegistrationResult with success status and column names
        """
        try:
            # Validate inputs
            await self._validate_registration_request(
                database, schema, table, text_column, embedding_config
            )
            
            # Get database manager
            db_manager = await self._get_database_manager(database)
            
            # Setup schema operations
            introspector = SchemaIntrospector(db_manager.pool)
            operations = SafeSchemaOperations(db_manager.pool, OperationMode.SAFE, timeout_seconds=300)
            collision_detector = ColumnCollisionDetector(db_manager.pool)
            
            # Check if table exists
            table_exists = await introspector.table_exists(schema, table)
            if not table_exists:
                return RegistrationResult(
                    success=False,
                    error=f"Table {schema}.{table} does not exist"
                )
            
            # Get table information
            table_info = await introspector.get_table_info(schema, table)
            
            # Validate text column exists
            if not table_info.has_column(text_column):
                return RegistrationResult(
                    success=False,
                    error=f"Text column '{text_column}' does not exist in table {schema}.{table}"
                )
            
            # Generate column names if not provided
            if vector_column is None:
                vector_column = f"{text_column}_vector"
            if model_column is None:
                model_column = f"{text_column}_embedding_model"
            
            timestamp_column = f"{vector_column}_created_at"
            
            # Get embedding configuration
            embedding_config_obj = self.config.embeddings.get(embedding_config)
            if not embedding_config_obj:
                return RegistrationResult(
                    success=False,
                    error=f"Embedding configuration '{embedding_config}' not found"
                )
            
            dimensions = embedding_config_obj.dimensions
            
            # Check for column collisions and resolve
            warnings = []
            vector_column = await self._resolve_column_collision(
                collision_detector, table_info, vector_column, "vector", warnings
            )
            model_column = await self._resolve_column_collision(
                collision_detector, table_info, model_column, "text", warnings
            )
            timestamp_column = await self._resolve_column_collision(
                collision_detector, table_info, timestamp_column, "timestamp", warnings
            )
            
            # Create columns if they don't exist
            changes_made = []
            
            # Add vector column
            if not table_info.has_column(vector_column):
                vector_change = await operations.create_vector_column(
                    schema, table, vector_column, dimensions
                )
                changes_made.append(f"Created vector column: {vector_column}")
            
            # Add model tracking column
            if not table_info.has_column(model_column):
                model_change = await operations.add_column_if_not_exists(
                    schema, table, model_column, "VARCHAR(255)", 
                    comment="Tracks which embedding model was used"
                )
                changes_made.append(f"Created model column: {model_column}")
            
            # Add timestamp column
            if not table_info.has_column(timestamp_column):
                timestamp_change = await operations.add_column_if_not_exists(
                    schema, table, timestamp_column, "TIMESTAMP", 
                    comment="When the vector was created"
                )
                changes_made.append(f"Created timestamp column: {timestamp_column}")
            
            # Create trigger for automatic vectorization
            await self._create_vectorization_trigger(
                db_manager, schema, table, text_column, vector_column
            )
            changes_made.append("Created vectorization trigger")
            
            logger.info(f"Successfully registered table {schema}.{table} for vectorization")
            
            if changes_made:
                warnings.extend([f"Changes made: {', '.join(changes_made)}"])
            
            return RegistrationResult(
                success=True,
                vector_column=vector_column,
                model_column=model_column,
                timestamp_column=timestamp_column,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Failed to register table {schema}.{table}: {e}")
            return RegistrationResult(
                success=False,
                error=str(e)
            )

    async def _validate_registration_request(
        self,
        database: str,
        schema: str,
        table: str,
        text_column: str,
        embedding_config: str,
    ) -> None:
        """Validate registration request parameters."""
        
        # Check database exists in config
        db_config = None
        for db in self.config.databases:
            if db.name == database:
                db_config = db
                break
        
        if not db_config:
            raise ValidationError(f"Database '{database}' not found in configuration")
        
        # Check embedding config exists
        if embedding_config not in self.config.embeddings:
            raise ValidationError(f"Embedding configuration '{embedding_config}' not found")
        
        # Basic name validation
        if not schema or not table or not text_column:
            raise ValidationError("Schema, table, and text_column are required")

    async def _get_database_manager(self, database_name: str) -> DatabaseManager:
        """Get or create database manager for the specified database."""
        if database_name not in self.database_managers:
            # Find database config
            db_config = None
            for db in self.config.databases:
                if db.name == database_name:
                    db_config = db
                    break
            
            if not db_config:
                raise ValidationError(f"Database '{database_name}' not found in configuration")
            
            # Create database manager
            manager = DatabaseManager(db_config.connection)
            self.database_managers[database_name] = manager
        
        return self.database_managers[database_name]

    async def _resolve_column_collision(
        self,
        collision_detector: ColumnCollisionDetector,
        table_info: Any,
        column_name: str,
        column_type: str,
        warnings: List[str],
    ) -> str:
        """Resolve column name collision if it exists."""
        if not table_info.has_column(column_name):
            return column_name
        
        # Column exists, check for collision
        collision_info = await collision_detector.check_single_collision(
            table_info.schema, table_info.name, column_name
        )
        
        if collision_info.severity == CollisionSeverity.CRITICAL:
            raise SchemaError(f"Critical collision: {collision_info.details}")
        
        # Generate alternative name if needed
        if collision_info.alternative_names:
            alternative = collision_info.alternative_names[0]
        else:
            # Fallback: simple suffix
            alternative = f"{column_name}_v2"
        
        warnings.append(f"Column '{column_name}' exists, using '{alternative}' instead")
        
        return alternative

    async def _create_vectorization_trigger(
        self,
        db_manager: DatabaseManager,
        schema: str,
        table: str,
        text_column: str,
        vector_column: str,
    ) -> None:
        """Create PostgreSQL trigger for automatic vectorization."""
        
        trigger_name = f"{table}_{text_column}_vector_trigger"
        channel_name = "v10r_events"
        config_key = f"{schema}.{table}.{text_column}"
        
        # Create trigger SQL
        trigger_sql = f"""
        CREATE OR REPLACE TRIGGER {trigger_name}
        AFTER INSERT OR UPDATE ON {schema}.{table}
        FOR EACH ROW
        WHEN (
            NEW.{vector_column} IS NULL 
            OR (TG_OP = 'UPDATE' AND OLD.{text_column} IS DISTINCT FROM NEW.{text_column})
        )
        EXECUTE FUNCTION v10r.generic_vector_notify(
            '{channel_name}',
            '{config_key}',
            '{vector_column.replace("_vector", "_embedding_model")}',
            '{text_column}'
        );
        """
        
        async with db_manager.pool.acquire() as conn:
            await conn.execute(trigger_sql)
        
        logger.info(f"Created trigger {trigger_name} for {schema}.{table}")

    async def close(self) -> None:
        """Close all database managers."""
        for manager in self.database_managers.values():
            await manager.close()


class RegistrationError(Exception):
    """Raised when table registration fails."""
    pass 