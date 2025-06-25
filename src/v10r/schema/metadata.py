"""
Metadata management for v10r.

Handles creation and management of v10r metadata schema
and tracking tables for column registry, migrations, etc.
"""

import logging
from typing import Dict, List, Optional, Any
from enum import Enum

from ..database.connection import ConnectionPool
from ..database.introspection import SchemaIntrospector
from ..exceptions import SchemaError, DatabaseError


logger = logging.getLogger(__name__)


class MetadataSchema(str, Enum):
    """v10r metadata schema names."""
    
    METADATA = "v10r_metadata"
    QA = "v10r_qa"
    CORE = "v10r"


class MetadataManager:
    """Manages v10r metadata schema and tables."""
    
    def __init__(self, pool: ConnectionPool):
        self.pool = pool
        self.introspector = SchemaIntrospector(pool)
        
        # Define required metadata tables
        self.required_tables = {
            "column_registry": self._get_column_registry_ddl(),
            "dimension_migrations": self._get_dimension_migrations_ddl(),
            "schema_drift_log": self._get_schema_drift_log_ddl(),
            "collision_log": self._get_collision_log_ddl(),
            "vectorization_log": self._get_vectorization_log_ddl(),
            "null_vector_status": self._get_null_vector_status_ddl(),
        }
        
        # QA and monitoring views
        self.qa_views = {
            "system_health": self._get_system_health_view_ddl(),
            "completion_summary": self._get_completion_summary_view_ddl(),
        }
        
        # Core functions
        self.core_functions = {
            "check_null_vectors": self._get_check_null_vectors_function_ddl(),
            "reconcile_schemas": self._get_reconcile_schemas_function_ddl(),
        }
    
    async def setup_metadata_schema(self) -> Dict[str, Any]:
        """Set up complete v10r metadata schema."""
        results = {
            "schemas_created": [],
            "tables_created": [],
            "views_created": [],
            "functions_created": [],
            "errors": [],
        }
        
        try:
            # Create schemas
            for schema in MetadataSchema:
                try:
                    await self._create_schema_if_not_exists(schema.value)
                    results["schemas_created"].append(schema.value)
                except Exception as e:
                    error_msg = f"Failed to create schema {schema.value}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Create metadata tables
            for table_name, ddl in self.required_tables.items():
                try:
                    await self._create_table_if_not_exists(
                        MetadataSchema.METADATA.value, table_name, ddl
                    )
                    results["tables_created"].append(f"{MetadataSchema.METADATA.value}.{table_name}")
                except Exception as e:
                    error_msg = f"Failed to create table {table_name}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Create QA views
            for view_name, ddl in self.qa_views.items():
                try:
                    await self._create_view_if_not_exists(
                        MetadataSchema.QA.value, view_name, ddl
                    )
                    results["views_created"].append(f"{MetadataSchema.QA.value}.{view_name}")
                except Exception as e:
                    error_msg = f"Failed to create view {view_name}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Create core functions
            for function_name, ddl in self.core_functions.items():
                try:
                    await self._create_function(MetadataSchema.CORE.value, function_name, ddl)
                    results["functions_created"].append(f"{MetadataSchema.CORE.value}.{function_name}")
                except Exception as e:
                    error_msg = f"Failed to create function {function_name}: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            logger.info(f"Metadata schema setup completed: {len(results['errors'])} errors")
            return results
            
        except Exception as e:
            logger.error(f"Metadata schema setup failed: {e}")
            raise SchemaError(f"Failed to setup metadata schema: {e}") from e
    
    async def check_metadata_integrity(self) -> Dict[str, Any]:
        """Check integrity of metadata schema."""
        integrity_report = {
            "schemas_exist": {},
            "tables_exist": {},
            "views_exist": {},
            "functions_exist": {},
            "missing_components": [],
            "is_healthy": True,
        }
        
        try:
            # Check schemas
            existing_schemas = await self.introspector.list_schemas()
            for schema in MetadataSchema:
                exists = schema.value in existing_schemas
                integrity_report["schemas_exist"][schema.value] = exists
                if not exists:
                    integrity_report["missing_components"].append(f"schema:{schema.value}")
                    integrity_report["is_healthy"] = False
            
            # Check tables
            if MetadataSchema.METADATA.value in existing_schemas:
                existing_tables = await self.introspector.list_tables(MetadataSchema.METADATA.value)
                table_names = [table for schema, table in existing_tables]
                
                for table_name in self.required_tables.keys():
                    exists = table_name in table_names
                    integrity_report["tables_exist"][table_name] = exists
                    if not exists:
                        integrity_report["missing_components"].append(f"table:{table_name}")
                        integrity_report["is_healthy"] = False
            
            # Check views
            if MetadataSchema.QA.value in existing_schemas:
                existing_qa_tables = await self.introspector.list_tables(MetadataSchema.QA.value)
                qa_names = [table for schema, table in existing_qa_tables]
                
                for view_name in self.qa_views.keys():
                    exists = view_name in qa_names
                    integrity_report["views_exist"][view_name] = exists
                    if not exists:
                        integrity_report["missing_components"].append(f"view:{view_name}")
                        # Views are not critical for health
            
            return integrity_report
            
        except Exception as e:
            logger.error(f"Metadata integrity check failed: {e}")
            return {
                "error": str(e),
                "is_healthy": False,
            }
    
    async def _create_schema_if_not_exists(self, schema_name: str) -> None:
        """Create schema if it doesn't exist."""
        sql = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.schemata 
                WHERE schema_name = '{schema_name}'
            ) THEN
                CREATE SCHEMA {schema_name};
                RAISE NOTICE 'Created schema %', '{schema_name}';
            ELSE
                RAISE NOTICE 'Schema % already exists', '{schema_name}';
            END IF;
        END $$;
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(sql)
    
    async def _create_table_if_not_exists(self, schema: str, table: str, ddl: str) -> None:
        """Create table if it doesn't exist."""
        check_sql = f"""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables 
            WHERE table_schema = '{schema}' AND table_name = '{table}'
        )
        """
        
        async with self.pool.acquire() as conn:
            exists = await conn.fetchval(check_sql)
            if not exists:
                await conn.execute(ddl)
                logger.info(f"Created table {schema}.{table}")
            else:
                logger.debug(f"Table {schema}.{table} already exists")
    
    async def _create_view_if_not_exists(self, schema: str, view: str, ddl: str) -> None:
        """Create view if it doesn't exist."""
        drop_sql = f"DROP VIEW IF EXISTS {schema}.{view};"
        
        async with self.pool.acquire() as conn:
            # Always recreate views to ensure they're up to date
            await conn.execute(drop_sql)
            await conn.execute(ddl)
            logger.info(f"Created/updated view {schema}.{view}")
    
    async def _create_function(self, schema: str, function: str, ddl: str) -> None:
        """Create or replace function."""
        async with self.pool.acquire() as conn:
            await conn.execute(ddl)
            logger.info(f"Created/updated function {schema}.{function}")
    
    # DDL definitions for metadata tables
    
    def _get_column_registry_ddl(self) -> str:
        """Get DDL for column registry table."""
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.column_registry (
            id SERIAL PRIMARY KEY,
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            original_column_name VARCHAR(255) NOT NULL,
            actual_column_name VARCHAR(255) NOT NULL,
            column_type VARCHAR(50) NOT NULL,
            dimension INTEGER,
            model_name VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW(),
            deprecated_at TIMESTAMP,
            replacement_column VARCHAR(255),
            config_key VARCHAR(255),
            
            CONSTRAINT unique_actual_column UNIQUE(database_name, schema_name, table_name, actual_column_name),
            CONSTRAINT valid_column_type CHECK (column_type IN ('vector', 'text', 'model', 'timestamp', 'cleaned'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_column_registry_table 
        ON v10r_metadata.column_registry(database_name, schema_name, table_name);
        
        CREATE INDEX IF NOT EXISTS idx_column_registry_config 
        ON v10r_metadata.column_registry(config_key);
        """
    
    def _get_dimension_migrations_ddl(self) -> str:
        """Get DDL for dimension migrations table."""
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.dimension_migrations (
            id SERIAL PRIMARY KEY,
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            old_column VARCHAR(255) NOT NULL,
            old_dimension INTEGER NOT NULL,
            new_column VARCHAR(255) NOT NULL,
            new_dimension INTEGER NOT NULL,
            migration_started_at TIMESTAMP DEFAULT NOW(),
            migration_completed_at TIMESTAMP,
            rows_migrated INTEGER DEFAULT 0,
            total_rows INTEGER,
            status VARCHAR(50) DEFAULT 'pending',
            
            CONSTRAINT valid_migration_status CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_dimension_migrations_table 
        ON v10r_metadata.dimension_migrations(database_name, schema_name, table_name);
        
        CREATE INDEX IF NOT EXISTS idx_dimension_migrations_status 
        ON v10r_metadata.dimension_migrations(status);
        """
    
    def _get_schema_drift_log_ddl(self) -> str:
        """Get DDL for schema drift log table."""
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.schema_drift_log (
            id SERIAL PRIMARY KEY,
            detected_at TIMESTAMP DEFAULT NOW(),
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            drift_type VARCHAR(50) NOT NULL,
            expected_state JSONB,
            actual_state JSONB,
            resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP,
            resolution_method VARCHAR(100),
            
            CONSTRAINT valid_drift_type CHECK (drift_type IN ('missing_column', 'wrong_type', 'dimension_mismatch', 'missing_index', 'missing_trigger'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_schema_drift_table 
        ON v10r_metadata.schema_drift_log(database_name, schema_name, table_name);
        
        CREATE INDEX IF NOT EXISTS idx_schema_drift_unresolved 
        ON v10r_metadata.schema_drift_log(resolved, detected_at) WHERE NOT resolved;
        """
    
    def _get_collision_log_ddl(self) -> str:
        """Get DDL for collision log table."""
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.collision_log (
            id SERIAL PRIMARY KEY,
            occurred_at TIMESTAMP DEFAULT NOW(),
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            desired_column_name VARCHAR(255) NOT NULL,
            existing_column_type VARCHAR(100),
            existing_column_info JSONB,
            collision_severity VARCHAR(20) NOT NULL,
            collision_type VARCHAR(50) NOT NULL,
            resolution_strategy VARCHAR(50),
            resolved_column_name VARCHAR(255),
            user_decision VARCHAR(255),
            alert_sent BOOLEAN DEFAULT FALSE,
            config_key VARCHAR(255),
            details TEXT,
            
            CONSTRAINT valid_collision_severity CHECK (collision_severity IN ('critical', 'warning', 'info', 'safe')),
            CONSTRAINT valid_collision_type CHECK (collision_type IN ('existing_vector', 'previous_vectorizer', 'vectorizer_pattern', 'unrelated_column', 'no_collision'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_collision_patterns 
        ON v10r_metadata.collision_log(schema_name, table_name, collision_type, occurred_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_collision_severity 
        ON v10r_metadata.collision_log(collision_severity, occurred_at DESC);
        """
    
    def _get_vectorization_log_ddl(self) -> str:
        """Get DDL for vectorization log table.""" 
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.vectorization_log (
            id SERIAL PRIMARY KEY,
            created_at TIMESTAMP DEFAULT NOW(),
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            row_id JSONB NOT NULL,
            text_column VARCHAR(255) NOT NULL,
            vector_column VARCHAR(255) NOT NULL,
            embedding_model VARCHAR(255),
            processing_time INTEGER, -- milliseconds
            status VARCHAR(20) NOT NULL,
            error_message TEXT,
            api_response_time INTEGER, -- milliseconds
            embedding_endpoint VARCHAR(500),
            text_length INTEGER,
            vector_dimension INTEGER,
            
            CONSTRAINT valid_vectorization_status CHECK (status IN ('success', 'failed', 'retry', 'skipped'))
        );
        
        CREATE INDEX IF NOT EXISTS idx_vectorization_log_table 
        ON v10r_metadata.vectorization_log(database_name, schema_name, table_name);
        
        CREATE INDEX IF NOT EXISTS idx_vectorization_log_status 
        ON v10r_metadata.vectorization_log(status, created_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_vectorization_log_performance 
        ON v10r_metadata.vectorization_log(embedding_model, processing_time) WHERE status = 'success';
        """
    
    def _get_null_vector_status_ddl(self) -> str:
        """Get DDL for null vector status table."""
        return """
        CREATE TABLE IF NOT EXISTS v10r_metadata.null_vector_status (
            id SERIAL PRIMARY KEY,
            checked_at TIMESTAMP DEFAULT NOW(),
            database_name VARCHAR(255) NOT NULL,
            schema_name VARCHAR(255) NOT NULL,
            table_name VARCHAR(255) NOT NULL,
            vector_column VARCHAR(255) NOT NULL,
            config_key VARCHAR(255),
            null_vectors INTEGER NOT NULL,
            total_rows INTEGER NOT NULL,
            completion_percentage DECIMAL(5,2) GENERATED ALWAYS AS (
                CASE 
                    WHEN total_rows > 0 THEN 
                        ROUND(((total_rows - null_vectors) * 100.0 / total_rows), 2)
                    ELSE 100.0 
                END
            ) STORED,
            last_vectorized_at TIMESTAMP,
            
            CONSTRAINT unique_vector_status UNIQUE(database_name, schema_name, table_name, vector_column, checked_at::date)
        );
        
        CREATE INDEX IF NOT EXISTS idx_null_vector_status_completion 
        ON v10r_metadata.null_vector_status(completion_percentage, checked_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_null_vector_status_nulls 
        ON v10r_metadata.null_vector_status(null_vectors, checked_at DESC) WHERE null_vectors > 0;
        """
    
    # QA Views
    
    def _get_system_health_view_ddl(self) -> str:
        """Get DDL for system health view."""
        return """
        CREATE VIEW v10r_qa.system_health AS
        SELECT 
            -- Completeness metrics
            COUNT(*) FILTER (WHERE nvs.vector_column IS NOT NULL) as vectors_populated,
            COUNT(*) FILTER (WHERE nvs.null_vectors > 0) as tables_with_nulls,
            AVG(nvs.completion_percentage) as avg_completion_percentage,
            
            -- Quality metrics 
            COUNT(DISTINCT vl.embedding_model) as unique_models,
            COUNT(*) FILTER (WHERE vl.status = 'failed') as failed_vectorizations,
            COUNT(*) FILTER (WHERE vl.status = 'success') as successful_vectorizations,
            
            -- Performance metrics
            AVG(vl.processing_time) FILTER (WHERE vl.status = 'success') as avg_processing_time_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY vl.processing_time) FILTER (WHERE vl.status = 'success') as p95_processing_time_ms,
            
            -- Drift and issues
            COUNT(*) FILTER (WHERE sdl.resolved = false) as unresolved_drift_issues,
            COUNT(*) FILTER (WHERE cl.collision_severity = 'critical') as critical_collisions,
            
            -- Timestamps
            MAX(nvs.checked_at) as last_null_check,
            MAX(vl.created_at) as last_vectorization,
            MAX(sdl.detected_at) as last_drift_detection
            
        FROM v10r_metadata.null_vector_status nvs
        LEFT JOIN v10r_metadata.vectorization_log vl ON 
            vl.database_name = nvs.database_name AND
            vl.schema_name = nvs.schema_name AND 
            vl.table_name = nvs.table_name AND
            vl.created_at > NOW() - INTERVAL '24 hours'
        LEFT JOIN v10r_metadata.schema_drift_log sdl ON
            sdl.database_name = nvs.database_name AND
            sdl.schema_name = nvs.schema_name AND
            sdl.table_name = nvs.table_name
        LEFT JOIN v10r_metadata.collision_log cl ON
            cl.database_name = nvs.database_name AND
            cl.schema_name = nvs.schema_name AND
            cl.table_name = nvs.table_name AND
            cl.occurred_at > NOW() - INTERVAL '7 days'
        WHERE nvs.checked_at > NOW() - INTERVAL '24 hours';
        """
    
    def _get_completion_summary_view_ddl(self) -> str:
        """Get DDL for completion summary view."""
        return """
        CREATE VIEW v10r_qa.completion_summary AS
        SELECT 
            database_name,
            schema_name,
            table_name,
            vector_column,
            config_key,
            null_vectors,
            total_rows,
            completion_percentage,
            last_vectorized_at,
            checked_at,
            CASE 
                WHEN completion_percentage >= 95 THEN 'healthy'
                WHEN completion_percentage >= 80 THEN 'warning'
                ELSE 'critical'
            END as health_status,
            CASE 
                WHEN last_vectorized_at < NOW() - INTERVAL '1 hour' THEN 'stale'
                WHEN last_vectorized_at < NOW() - INTERVAL '10 minutes' THEN 'recent'
                ELSE 'current'
            END as freshness_status
        FROM v10r_metadata.null_vector_status nvs
        WHERE checked_at = (
            SELECT MAX(checked_at) 
            FROM v10r_metadata.null_vector_status nvs2 
            WHERE nvs2.database_name = nvs.database_name
            AND nvs2.schema_name = nvs.schema_name
            AND nvs2.table_name = nvs.table_name
            AND nvs2.vector_column = nvs.vector_column
        )
        ORDER BY completion_percentage ASC, null_vectors DESC;
        """
    
    # Core Functions
    
    def _get_check_null_vectors_function_ddl(self) -> str:
        """Get DDL for check null vectors function."""
        return """
        CREATE OR REPLACE FUNCTION v10r.check_null_vectors(
            p_database text,
            p_schema text,
            p_table text,
            p_vector_column text
        )
        RETURNS integer AS $$
        DECLARE
            v_null_count integer;
            v_total_count integer;
        BEGIN
            -- Count NULL vectors and total rows
            EXECUTE format(
                'SELECT COUNT(*) FILTER (WHERE %I IS NULL), COUNT(*) FROM %I.%I',
                p_vector_column, p_schema, p_table
            ) INTO v_null_count, v_total_count;
            
            -- Update null vector status
            INSERT INTO v10r_metadata.null_vector_status (
                database_name, schema_name, table_name, vector_column,
                null_vectors, total_rows, checked_at
            ) VALUES (
                p_database, p_schema, p_table, p_vector_column,
                v_null_count, v_total_count, NOW()
            )
            ON CONFLICT (database_name, schema_name, table_name, vector_column, (checked_at::date))
            DO UPDATE SET
                null_vectors = EXCLUDED.null_vectors,
                total_rows = EXCLUDED.total_rows,
                checked_at = EXCLUDED.checked_at;
            
            -- Trigger bulk vectorization if needed
            IF v_null_count > 0 THEN
                PERFORM pg_notify('v10r_bulk_vectorize', json_build_object(
                    'database', p_database,
                    'schema', p_schema,
                    'table', p_table,
                    'vector_column', p_vector_column,
                    'null_count', v_null_count,
                    'priority', 'low'
                )::text);
            END IF;
            
            RETURN v_null_count;
        END;
        $$ LANGUAGE plpgsql;
        """
    
    def _get_reconcile_schemas_function_ddl(self) -> str:
        """Get DDL for schema reconciliation trigger function."""
        return """
        CREATE OR REPLACE FUNCTION v10r.reconcile_schemas()
        RETURNS void AS $$
        BEGIN
            -- Notify v10r service to check schema drift
            PERFORM pg_notify('v10r_reconcile', json_build_object(
                'action', 'reconcile_all',
                'timestamp', extract(epoch from now()),
                'database', current_database()
            )::text);
            
            RAISE NOTICE 'Schema reconciliation request sent';
        END;
        $$ LANGUAGE plpgsql;
        """ 