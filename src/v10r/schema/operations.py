"""
Safe schema operations for v10r.

Provides safe ALTER TABLE operations with rollback capabilities,
idempotent patterns, and production-ready safety checks.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, AsyncIterator
from enum import Enum

from ..database.connection import ConnectionPool
from ..database.introspection import SchemaIntrospector, ColumnInfo
from ..exceptions import SchemaError, DatabaseError


logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of schema changes."""
    
    ADD_COLUMN = "add_column"
    ADD_VECTOR_COLUMN = "add_vector_column"
    MODIFY_COLUMN = "modify_column"
    DROP_COLUMN = "drop_column"
    CREATE_INDEX = "create_index"
    ADD_CONSTRAINT = "add_constraint"


class OperationMode(str, Enum):
    """Schema operation modes."""
    
    SAFE = "safe"              # Maximum safety checks, no destructive operations
    FORCE = "force"            # Allow destructive operations with warnings
    DRY_RUN = "dry_run"        # Generate SQL but don't execute
    PRODUCTION = "production"   # Production mode with all safety features


@dataclass
class SchemaChange:
    """Represents a schema change operation."""
    
    change_type: ChangeType
    schema: str
    table: str
    description: str
    sql: str
    rollback_sql: Optional[str] = None
    
    # Legacy fields for backward compatibility
    target_object: Optional[str] = None  # Column name, index name, etc.
    old_definition: Optional[str] = None
    new_definition: Optional[str] = None
    sql_commands: List[str] = None
    
    # Safety metadata
    is_destructive: bool = False
    requires_lock: bool = False
    estimated_duration_seconds: Optional[float] = None
    rollback_commands: List[str] = None
    
    # Execution results
    executed: bool = False
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.sql_commands is None:
            self.sql_commands = []
        if self.rollback_commands is None:
            self.rollback_commands = []
    
    @property
    def full_table_name(self) -> str:
        """Get fully qualified table name."""
        return f"{self.schema}.{self.table}"
    
    @property 
    def table_full_name(self) -> str:
        """Alias for full_table_name (for test compatibility)."""
        return self.full_table_name
    
    @property
    def is_executed(self) -> bool:
        """Check if this change has been executed."""
        return self.executed
    
    @property
    def can_rollback(self) -> bool:
        """Check if this change can be rolled back."""
        return self.rollback_sql is not None and len(self.rollback_sql.strip()) > 0
    
    @property
    def has_error(self) -> bool:
        """Check if this change has an error."""
        return self.error is not None
    
    @property
    def change_id(self) -> str:
        """Get unique identifier for this change."""
        target = self.target_object or "unknown"
        return f"{self.change_type.value}_{self.schema}_{self.table}_{target}"


class SafeSchemaOperations:
    """Safe schema operation executor with rollback capabilities."""
    
    def __init__(
        self,
        pool: ConnectionPool,
        operation_mode: OperationMode = OperationMode.SAFE,
        timeout_seconds: int = 300,
    ):
        self.pool = pool
        self.operation_mode = operation_mode
        self.mode = operation_mode  # Alias for backward compatibility
        self.timeout_seconds = timeout_seconds
        self.introspector = SchemaIntrospector(pool)
        
        # Safety limits
        self.max_concurrent_operations = 1
        self.max_table_size_for_alter = 1_000_000  # rows
        self.backup_before_destructive = True
        
        # Track active operations
        self._active_operations: Dict[str, SchemaChange] = {}
        self._operation_lock = asyncio.Lock()
    
    async def add_column_if_not_exists(
        self,
        schema: str,
        table: str,
        column_name: str,
        column_type: str,
        **kwargs,
    ) -> SchemaChange:
        """
        Add a column if it doesn't exist using idempotent pattern.
        
        Args:
            schema: Target schema
            table: Target table
            column_name: Column name to add
            column_type: PostgreSQL column type
            **kwargs: Additional column options (default, not_null, etc.)
        """
        # Build column definition
        definition_parts = [column_name, column_type]
        
        if kwargs.get("not_null"):
            definition_parts.append("NOT NULL")
        
        if "default" in kwargs:
            definition_parts.append(f"DEFAULT {kwargs['default']}")
        
        column_definition = " ".join(definition_parts)
        
        # Create idempotent SQL
        sql_commands = [
            f"""
            DO $$
            BEGIN
                -- Check if column exists
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table}' 
                    AND column_name = '{column_name}'
                ) THEN
                    -- Add column with safety checks
                    ALTER TABLE {schema}.{table} 
                    ADD COLUMN {column_definition};
                    
                    RAISE NOTICE 'Added column % to %.%', 
                        '{column_name}', '{schema}', '{table}';
                ELSE
                    RAISE NOTICE 'Column % already exists in %.%', 
                        '{column_name}', '{schema}', '{table}';
                END IF;
            END $$;
            """
        ]
        
        # Generate rollback command to drop the column
        rollback_commands = [
            f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS {column_name};"
        ]
        
        change = SchemaChange(
            change_type=ChangeType.ADD_COLUMN,
            schema=schema,
            table=table,
            description=f"Add column {column_name}",
            sql=sql_commands[0] if sql_commands else "",
            rollback_sql=rollback_commands[0] if rollback_commands else None,
            target_object=column_name,
            new_definition=column_definition,
            sql_commands=sql_commands,
            rollback_commands=rollback_commands,
            is_destructive=False,
            requires_lock=True,
        )
        
        return await self._execute_change(change)
    
    async def create_vector_column(
        self,
        schema: str,
        table: str,
        column_name: str,
        dimension: int,
    ) -> SchemaChange:
        """Create a pgvector column with specific dimension."""
        change = await self.add_column_if_not_exists(
            schema=schema,
            table=table,
            column_name=column_name,
            column_type=f"vector({dimension})",
        )
        # Update change type to reflect vector column creation
        change.change_type = ChangeType.ADD_VECTOR_COLUMN
        change.description = f"Create pgvector column {column_name} with dimension {dimension}"
        return change
    
    async def create_index(
        self,
        schema: str,
        table: str,
        index_name: str,
        columns: List[str],
        index_type: str = "btree",
        **kwargs,
    ) -> SchemaChange:
        """Create an index (alias for create_index_if_not_exists)."""
        return await self.create_index_if_not_exists(
            schema=schema,
            table=table,
            index_name=index_name,
            columns=columns,
            index_type=index_type,
            **kwargs,
        )
    
    async def create_index_if_not_exists(
        self,
        schema: str,
        table: str,
        index_name: str,
        columns: List[str],
        index_type: str = "btree",
        **kwargs,
    ) -> SchemaChange:
        """Create an index if it doesn't exist."""
        
        # Build index definition
        columns_str = ", ".join(columns)
        index_options = []
        
        if kwargs.get("unique"):
            index_options.append("UNIQUE")
        
        if kwargs.get("concurrent", True):
            index_options.append("CONCURRENTLY")
        
        options_str = " ".join(index_options)
        
        # For vector indexes
        using_clause = f"USING {index_type}"
        if index_type in ("ivfflat", "hnsw"):
            using_clause = f"USING {index_type}"
        
        sql_commands = [
            f"""
            DO $$
            BEGIN
                -- Check if index exists
                IF NOT EXISTS (
                    SELECT 1 FROM pg_indexes 
                    WHERE schemaname = '{schema}' 
                    AND tablename = '{table}' 
                    AND indexname = '{index_name}'
                ) THEN
                    -- Create index
                    CREATE {options_str} INDEX {index_name}
                    ON {schema}.{table} {using_clause} ({columns_str});
                    
                    RAISE NOTICE 'Created index % on %.%', 
                        '{index_name}', '{schema}', '{table}';
                ELSE
                    RAISE NOTICE 'Index % already exists on %.%', 
                        '{index_name}', '{schema}', '{table}';
                END IF;
            EXCEPTION WHEN OTHERS THEN
                -- Log error but don't fail
                RAISE NOTICE 'Could not create index %: %', 
                    '{index_name}', SQLERRM;
            END $$;
            """
        ]
        
        rollback_commands = [
            f"DROP INDEX IF EXISTS {schema}.{index_name};"
        ]
        
        change = SchemaChange(
            change_type=ChangeType.CREATE_INDEX,
            schema=schema,
            table=table,
            description=f"Create index {index_name}",
            sql=sql_commands[0] if sql_commands else "",
            rollback_sql=rollback_commands[0] if rollback_commands else None,
            target_object=index_name,
            new_definition=f"{index_type} on ({columns_str})",
            sql_commands=sql_commands,
            rollback_commands=rollback_commands,
            is_destructive=False,
            requires_lock=False,  # CONCURRENTLY doesn't require lock
        )
        
        return await self._execute_change(change)
    
    async def handle_dimension_change(
        self,
        schema: str,
        table: str,
        old_column: str,
        old_dimension: int,
        new_column: str,
        new_dimension: int,
    ) -> SchemaChange:
        """Handle vector dimension changes by creating new column."""
        
        sql_commands = [
            # Create new column
            f"""
            DO $$
            BEGIN
                -- Create new vector column with new dimension
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = '{schema}' 
                    AND table_name = '{table}' 
                    AND column_name = '{new_column}'
                ) THEN
                    ALTER TABLE {schema}.{table} 
                    ADD COLUMN {new_column} vector({new_dimension});
                    
                    RAISE NOTICE 'Created new vector column % with dimension %', 
                        '{new_column}', {new_dimension};
                END IF;
            END $$;
            """,
            
            # Add metadata tracking
            f"""
            INSERT INTO v10r_metadata.dimension_migrations (
                database_name,
                schema_name,
                table_name,
                old_column,
                old_dimension,
                new_column,
                new_dimension,
                migration_started_at,
                status
            ) VALUES (
                current_database(),
                '{schema}',
                '{table}',
                '{old_column}',
                {old_dimension},
                '{new_column}',
                {new_dimension},
                NOW(),
                'pending'
            )
            ON CONFLICT DO NOTHING;
            """
        ]
        
        rollback_commands = [
            f"ALTER TABLE {schema}.{table} DROP COLUMN IF EXISTS {new_column};",
            f"""
            DELETE FROM v10r_metadata.dimension_migrations 
            WHERE database_name = current_database()
            AND schema_name = '{schema}'
            AND table_name = '{table}'
            AND new_column = '{new_column}';
            """
        ]
        
        change = SchemaChange(
            change_type=ChangeType.ADD_COLUMN,  # Use supported enum value
            schema=schema,
            table=table,
            description=f"Dimension migration: {old_column} -> {new_column}",
            sql=sql_commands[0] if sql_commands else "",
            rollback_sql=rollback_commands[0] if rollback_commands else None,
            target_object=new_column,
            old_definition=f"{old_column} vector({old_dimension})",
            new_definition=f"{new_column} vector({new_dimension})",
            sql_commands=sql_commands,
            rollback_commands=rollback_commands,
            is_destructive=False,
            requires_lock=True,
        )
        
        return await self._execute_change(change)
    
    async def _execute_change(self, change: SchemaChange) -> SchemaChange:
        """Execute a schema change with safety checks and rollback support."""
        
        if self.operation_mode == OperationMode.DRY_RUN:
            change.executed = False
            change.description = f"DRY RUN: {change.description}"
            logger.info(f"DRY RUN: Would execute {change.change_id}")
            for sql in change.sql_commands:
                logger.info(f"SQL: {sql.strip()}")
            return change
        
        async with self._operation_lock:
            # Check concurrency limits
            if len(self._active_operations) >= self.max_concurrent_operations:
                raise SchemaError(
                    f"Too many concurrent operations ({len(self._active_operations)}). "
                    f"Maximum allowed: {self.max_concurrent_operations}"
                )
            
            # Pre-execution safety checks
            await self._pre_execution_checks(change)
            
            # Track operation
            self._active_operations[change.change_id] = change
            
            try:
                # Execute with timeout and transaction support
                await self._execute_with_transaction(change)
                change.executed = True
                logger.info(f"Successfully executed {change.change_id}")
                
            except Exception as e:
                change.executed = False  # Set executed to False on failure
                change.error = str(e)
                logger.error(f"Failed to execute {change.change_id}: {e}")
                
                # Attempt rollback if configured
                if change.rollback_commands and self.operation_mode != OperationMode.DRY_RUN:
                    await self._attempt_rollback(change)
                
                raise
            finally:
                # Remove from active operations
                self._active_operations.pop(change.change_id, None)
        
        return change
    
    async def _pre_execution_checks(self, change: SchemaChange) -> None:
        """Perform safety checks before executing a change."""
        
        # Check if table exists
        table_exists = await self.introspector.table_exists(change.schema, change.table)
        if not table_exists:
            raise SchemaError(f"Table {change.full_table_name} does not exist")
        
        # Check table size for potentially slow operations
        if change.requires_lock:
            table_info = await self.introspector.get_table_info(change.schema, change.table)
            if table_info and table_info.row_count:
                if table_info.row_count > self.max_table_size_for_alter:
                    if self.operation_mode == OperationMode.SAFE:
                        raise SchemaError(
                            f"Table {change.full_table_name} has {table_info.row_count} rows, "
                            f"which exceeds the safe limit of {self.max_table_size_for_alter}. "
                            f"Use FORCE mode to proceed."
                        )
                    else:
                        logger.warning(
                            f"Large table operation: {change.full_table_name} has "
                            f"{table_info.row_count} rows. This may take time."
                        )
        
        # Check for destructive operations in safe mode
        if change.is_destructive and self.operation_mode == OperationMode.SAFE:
            raise SchemaError(
                f"Destructive operation {change.change_type.value} not allowed in SAFE mode. "
                f"Use FORCE mode to proceed."
            )
    
    async def _execute_with_transaction(self, change: SchemaChange) -> None:
        """Execute schema change within a transaction with timeout."""
        
        start_time = time.time()
        
        # Determine which SQL to execute - prefer sql_commands if available, otherwise use sql
        sql_to_execute = change.sql_commands if change.sql_commands else [change.sql] if change.sql else []
        
        async with self.pool.acquire() as conn:
            # Set statement timeout
            await conn.execute(f"SET statement_timeout = '{self.timeout_seconds}s'")
            
            # Execute within transaction
            async with await conn.transaction():
                for sql_command in sql_to_execute:
                    await conn.execute(sql_command)
        
        change.execution_time_ms = (time.time() - start_time) * 1000
    
    async def _attempt_rollback(self, change: SchemaChange) -> None:
        """Attempt to rollback a failed operation."""
        try:
            logger.info(f"Attempting rollback for {change.change_id}")
            
            async with self.pool.acquire() as conn:
                async with await conn.transaction():
                    for rollback_command in change.rollback_commands:
                        await conn.execute(rollback_command)
            
            logger.info(f"Successfully rolled back {change.change_id}")
            
        except Exception as rollback_error:
            logger.error(f"Rollback failed for {change.change_id}: {rollback_error}")
            # Don't raise rollback errors, original error is more important
    
    async def execute_single(self, change: SchemaChange) -> SchemaChange:
        """Execute a single schema change with consistent error handling."""
        try:
            return await self._execute_change(change)
        except Exception as e:
            # Ensure error state is properly set even if _execute_change didn't handle it
            change.error = str(e)
            change.executed = False
            return change
    
    async def execute_batch(self, changes: List[SchemaChange]) -> List[SchemaChange]:
        """Execute multiple schema changes in sequence."""
        results = []
        
        for change in changes:
            try:
                result = await self._execute_change(change)
                results.append(result)
                
                # Stop on first failure in safe mode
                if not result.executed and self.operation_mode == OperationMode.SAFE:
                    logger.error(f"Stopping batch execution due to failure: {change.change_id}")
                    break
                    
            except Exception as e:
                change.error = str(e)
                change.executed = False
                results.append(change)
                
                if self.operation_mode == OperationMode.SAFE:
                    break
        
        return results
    
    async def rollback_single(self, change: SchemaChange) -> bool:
        """Rollback a single schema change."""
        if not change.can_rollback:
            change.error = "No rollback SQL available"
            return False
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(change.rollback_sql)
            change.executed = False
            change.error = None
            return True
        except Exception as e:
            change.error = f"Rollback failed: {e}"
            return False
    
    async def rollback_batch(self, changes: List[SchemaChange]) -> List[bool]:
        """Rollback multiple schema changes."""
        results = []
        
        # Rollback in reverse order
        for change in reversed(changes):
            result = await self.rollback_single(change)
            results.insert(0, result)  # Insert at beginning to maintain original order
        
        return results
    
    def _generate_rollback_sql(
        self, change_type: ChangeType, schema: str, table: str, **kwargs
    ) -> Optional[str]:
        """Generate rollback SQL for different change types."""
        if change_type == ChangeType.ADD_COLUMN:
            column = kwargs.get("column")
            if column:
                return f"ALTER TABLE {schema}.{table} DROP COLUMN {column};"
        elif change_type == ChangeType.CREATE_INDEX:
            index_name = kwargs.get("index", kwargs.get("index_name"))
            if index_name:
                return f"DROP INDEX IF EXISTS {schema}.{index_name};"
        elif change_type == ChangeType.ADD_CONSTRAINT:
            constraint_name = kwargs.get("constraint_name")
            if constraint_name:
                return f"ALTER TABLE {schema}.{table} DROP CONSTRAINT {constraint_name};"
        
        return None
    
    async def get_execution_summary(self, changes: List[SchemaChange]) -> Dict[str, Any]:
        """Get summary of execution results."""
        total = len(changes)
        successful = sum(1 for c in changes if c.executed)
        failed = sum(1 for c in changes if c.error)
        total_time = sum(c.execution_time_ms or 0 for c in changes)
        
        return {
            "total_operations": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_execution_time_ms": total_time,
            "failed_operations": [
                {
                    "change_id": c.change_id,
                    "error": c.error,
                    "change_type": c.change_type.value,
                }
                for c in changes if c.error
            ],
        } 