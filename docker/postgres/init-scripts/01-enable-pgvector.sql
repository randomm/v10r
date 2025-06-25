-- v10r Test Database Initialization Script
-- This script runs automatically when the PostgreSQL container starts

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create v10r metadata schema for tracking
CREATE SCHEMA IF NOT EXISTS v10r_metadata;

-- Create v10r QA schema for monitoring
CREATE SCHEMA IF NOT EXISTS v10r_qa;

-- Create v10r functions schema
CREATE SCHEMA IF NOT EXISTS v10r;

-- Grant permissions to test user
GRANT USAGE ON SCHEMA v10r_metadata TO v10r_test;
GRANT USAGE ON SCHEMA v10r_qa TO v10r_test;
GRANT USAGE ON SCHEMA v10r TO v10r_test;
GRANT CREATE ON SCHEMA v10r_metadata TO v10r_test;
GRANT CREATE ON SCHEMA v10r_qa TO v10r_test;
GRANT CREATE ON SCHEMA v10r TO v10r_test;

-- Create metadata tables for column tracking
CREATE TABLE IF NOT EXISTS v10r_metadata.column_registry (
    id SERIAL PRIMARY KEY,
    database_name VARCHAR(255),
    schema_name VARCHAR(255),
    table_name VARCHAR(255),
    original_column_name VARCHAR(255),
    actual_column_name VARCHAR(255),
    column_type VARCHAR(50),
    dimension INTEGER,
    model_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    deprecated_at TIMESTAMP,
    replacement_column VARCHAR(255),
    config_key VARCHAR(255),
    UNIQUE(database_name, schema_name, table_name, actual_column_name)
);

-- Create collision log table
CREATE TABLE IF NOT EXISTS v10r_metadata.collision_log (
    id SERIAL PRIMARY KEY,
    occurred_at TIMESTAMP DEFAULT NOW(),
    database_name VARCHAR(255),
    schema_name VARCHAR(255),
    table_name VARCHAR(255),
    desired_column_name VARCHAR(255),
    existing_column_type VARCHAR(100),
    existing_column_info JSONB,
    collision_severity VARCHAR(20),
    collision_type VARCHAR(50),
    resolution_strategy VARCHAR(50),
    resolved_column_name VARCHAR(255),
    user_decision VARCHAR(255),
    alert_sent BOOLEAN DEFAULT FALSE,
    config_key VARCHAR(255),
    details TEXT
);

-- Create dimension migrations table
CREATE TABLE IF NOT EXISTS v10r_metadata.dimension_migrations (
    id SERIAL PRIMARY KEY,
    database_name VARCHAR(255),
    schema_name VARCHAR(255),
    table_name VARCHAR(255),
    old_column VARCHAR(255),
    old_dimension INTEGER,
    new_column VARCHAR(255),
    new_dimension INTEGER,
    migration_started_at TIMESTAMP DEFAULT NOW(),
    migration_completed_at TIMESTAMP,
    rows_migrated INTEGER DEFAULT 0,
    total_rows INTEGER,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Create vectorization log for monitoring
CREATE TABLE IF NOT EXISTS v10r_metadata.vectorization_log (
    id SERIAL PRIMARY KEY,
    database_name VARCHAR(255),
    schema_name VARCHAR(255),
    table_name VARCHAR(255),
    row_id JSONB,
    text_column VARCHAR(255),
    vector_column VARCHAR(255),
    embedding_model VARCHAR(255),
    processing_time INTEGER, -- milliseconds
    api_response_time INTEGER, -- milliseconds
    embedding_endpoint VARCHAR(500),
    status VARCHAR(50), -- success, failed, retry
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Create pending tasks queue table
CREATE TABLE IF NOT EXISTS v10r_metadata.pending_tasks (
    id SERIAL PRIMARY KEY,
    task_id UUID DEFAULT gen_random_uuid(),
    database_name VARCHAR(255),
    schema_name VARCHAR(255),
    table_name VARCHAR(255),
    row_id JSONB,
    config_key VARCHAR(255),
    priority VARCHAR(20) DEFAULT 'medium', -- high, medium, low
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP DEFAULT NOW(),
    scheduled_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50) DEFAULT 'pending' -- pending, processing, completed, failed
);

-- Create indexes for performance
CREATE INDEX idx_collision_patterns ON v10r_metadata.collision_log(
    schema_name, table_name, collision_type, occurred_at DESC
);

CREATE INDEX idx_vectorization_log_status ON v10r_metadata.vectorization_log(
    status, created_at DESC
);

CREATE INDEX idx_pending_tasks_status ON v10r_metadata.pending_tasks(
    status, priority, scheduled_at
);

-- Create generic trigger function for vectorization events
CREATE OR REPLACE FUNCTION v10r.generic_vector_notify() RETURNS trigger AS $$
DECLARE
    channel_name TEXT;
    payload JSONB;
    model_column TEXT;
    needs_update BOOLEAN := FALSE;
BEGIN
    -- Get model column name from trigger arguments
    model_column := COALESCE(TG_ARGV[2], 'embedding_model');
    
    -- Check if update is needed
    IF TG_OP = 'INSERT' THEN
        needs_update := TRUE;
    ELSIF TG_OP = 'UPDATE' THEN
        -- Update if text changed or model is outdated
        IF OLD IS DISTINCT FROM NEW THEN
            -- Check if the text column changed (column name in TG_ARGV[3])
            EXECUTE format('SELECT ($1).%I IS DISTINCT FROM ($2).%I', 
                          TG_ARGV[3], TG_ARGV[3])
            INTO needs_update
            USING OLD, NEW;
        END IF;
    END IF;
    
    IF needs_update THEN
        -- Build channel name from table info
        channel_name := TG_ARGV[0];  -- Passed as trigger argument
        
        -- Build payload
        payload := jsonb_build_object(
            'database', current_database(),
            'schema', TG_TABLE_SCHEMA,
            'table', TG_TABLE_NAME,
            'operation', TG_OP,
            'id', to_jsonb(NEW),  -- Send full row for flexibility
            'old_id', CASE WHEN TG_OP = 'UPDATE' THEN to_jsonb(OLD) ELSE NULL END,
            'config_key', TG_ARGV[1],  -- Reference to embedding config
            'model_column', model_column  -- Column to store model info
        );
        
        -- Send notification
        PERFORM pg_notify(channel_name, payload::text);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions on the function
GRANT EXECUTE ON FUNCTION v10r.generic_vector_notify() TO v10r_test;

-- Create QA monitoring views
CREATE OR REPLACE VIEW v10r_qa.system_health AS
SELECT 
    COUNT(*) as total_vectorizations,
    COUNT(*) FILTER (WHERE status = 'success') as successful,
    COUNT(*) FILTER (WHERE status = 'failed') as failed,
    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour') as last_hour,
    AVG(processing_time) FILTER (WHERE status = 'success') as avg_processing_time_ms,
    MAX(created_at) as last_vectorization
FROM v10r_metadata.vectorization_log
WHERE created_at > NOW() - INTERVAL '24 hours';

-- Grant read permissions on views
GRANT SELECT ON v10r_qa.system_health TO v10r_test;

-- Log initialization completion
INSERT INTO v10r_metadata.vectorization_log (
    database_name, schema_name, table_name, status, created_at, completed_at
) VALUES (
    current_database(), 'v10r_metadata', 'init_complete', 'success', NOW(), NOW()
); 