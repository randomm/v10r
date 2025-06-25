"""
Integration tests for v10r using Docker Compose test environment.

Run with: pytest tests/integration/ -m integration

These tests use Docker Compose fixtures from conftest.py and don't require
any shell scripts - just run pytest and everything is handled automatically.
"""

import asyncio
import json
import pytest
import asyncpg
import redis
import httpx
from typing import Dict, List, Any, Optional


@pytest.mark.integration
class TestDockerEnvironment:
    """Test the Docker Compose test environment setup."""
    
    async def test_postgres_connection(self, test_db_connection):
        """Test PostgreSQL connection and pgvector extension."""
        # Test basic connection
        result = await test_db_connection.fetchval("SELECT 1")
        assert result == 1
        
        # Test pgvector extension
        result = await test_db_connection.fetchval(
            "SELECT extname FROM pg_extension WHERE extname = 'vector'"
        )
        assert result == "vector"
        
        # Test v10r schemas
        schemas = await test_db_connection.fetch(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE 'v10r%'"
        )
        schema_names = [s['schema_name'] for s in schemas]
        assert 'v10r_metadata' in schema_names
        assert 'v10r_qa' in schema_names
        assert 'v10r' in schema_names
    
    async def test_redis_connection(self, test_redis_connection):
        """Test Redis connection and basic operations."""
        # Test ping
        result = await test_redis_connection.ping()
        assert result is True
        
        # Test basic operations
        await test_redis_connection.set("test_key", "test_value")
        result = await test_redis_connection.get("test_key")
        assert result == "test_value"
        
        # Cleanup
        await test_redis_connection.delete("test_key")
    
    async def test_mock_infinity_server(self, mock_infinity_url):
        """Test mock Infinity server responses."""
        async with httpx.AsyncClient() as client:
            # Test health endpoint
            response = await client.get("http://localhost:18080/health")
            assert response.status_code == 200
            
            # Test embeddings endpoint
            response = await client.post(
                mock_infinity_url,
                json={
                    "model": "BAAI/bge-m3",
                    "input": ["test text"]
                }
            )
            assert response.status_code == 200
            
            data = response.json()
            assert "data" in data
            assert len(data["data"]) == 1
            assert "embedding" in data["data"][0]
            assert len(data["data"][0]["embedding"]) == 1024  # BGE-M3 dimensions


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test database integration features."""
    
    async def test_sample_data_insertion(self, test_db_connection, clean_test_data):
        """Test that sample data can be inserted and retrieved."""
        # Insert test article
        await test_db_connection.execute("""
            INSERT INTO articles (title, content, author) 
            VALUES ($1, $2, $3)
        """, "Test Article", "This is test content", "Test Author")
        
        # Retrieve and verify
        result = await test_db_connection.fetchrow(
            "SELECT title, content, author FROM articles WHERE title = $1",
            "Test Article"
        )
        
        assert result['title'] == "Test Article"
        assert result['content'] == "This is test content"
        assert result['author'] == "Test Author"
    
    async def test_vector_column_operations(self, test_db_connection, clean_test_data):
        """Test vector column operations with pgvector."""
        # Insert a vector
        test_vector = [0.1, 0.2, 0.3] * 256  # 768 dimensions
        # Convert to pgvector string format
        vector_str = f"[{','.join(map(str, test_vector))}]"
        
        await test_db_connection.execute("""
            INSERT INTO articles (title, content, title_vector) 
            VALUES ($1, $2, $3::vector)
        """, "Vector Test", "Content", vector_str)
        
        # Retrieve and verify vector
        result = await test_db_connection.fetchrow(
            "SELECT title_vector::text FROM articles WHERE title = $1",
            "Vector Test"
        )
        
        # pgvector returns vectors as strings like '[0.1,0.2,0.3,...]'
        retrieved_vector_str = result['title_vector']
        # Parse the vector string
        vector_values = [float(x) for x in retrieved_vector_str.strip('[]').split(',')]
        
        assert len(vector_values) == 768
        assert vector_values[:3] == [0.1, 0.2, 0.3]
    
    async def test_trigger_system(self, test_db_connection, clean_test_data):
        """Test that triggers are properly set up (if any exist)."""
        # Check if any triggers exist on our test tables
        triggers = await test_db_connection.fetch("""
            SELECT trigger_name, event_object_table 
            FROM information_schema.triggers 
            WHERE event_object_schema = 'public'
            AND event_object_table IN ('articles', 'products', 'user_profiles')
        """)
        
        # For now, just verify the query works
        # Later we'll add actual trigger tests when the trigger system is implemented
        assert isinstance(triggers, list)


@pytest.mark.integration  
class TestEmbeddingAPIIntegration:
    """Test integration with mock embedding APIs."""
    
    async def test_mock_infinity_embedding_generation(self, mock_infinity_url):
        """Test generating embeddings via mock Infinity server."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                mock_infinity_url,
                json={
                    "model": "BAAI/bge-m3",
                    "input": [
                        "First test document",
                        "Second test document",
                        "Third test document"
                    ]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "data" in data
            assert len(data["data"]) == 3
            
            for i, embedding_data in enumerate(data["data"]):
                assert embedding_data["index"] == i
                assert "embedding" in embedding_data
                assert len(embedding_data["embedding"]) == 1024
    
    async def test_batch_embedding_requests(self, mock_infinity_url):
        """Test handling of batch embedding requests."""
        # Create a larger batch to test batching logic
        texts = [f"Document {i} for batch testing" for i in range(50)]
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                mock_infinity_url,
                json={
                    "model": "BAAI/bge-m3", 
                    "input": texts
                },
                timeout=30  # Longer timeout for batch
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["data"]) == 50
            for i, item in enumerate(data["data"]):
                assert item["index"] == i
                assert len(item["embedding"]) == 1024


@pytest.mark.integration
class TestQueueIntegration:
    """Test message queue integration."""
    
    async def test_redis_task_queuing(self, test_redis_connection, clean_test_data):
        """Test queuing and processing tasks via Redis."""
        # Simulate queuing a vectorization task
        task = {
            "database": "v10r_test_db",
            "schema": "public",
            "table": "articles",
            "row_id": 1,
            "text_column": "content",
            "vector_column": "content_vector",
            "text": "This is content to be vectorized"
        }
        
        # Queue the task
        await test_redis_connection.lpush("v10r:tasks", json.dumps(task))
        
        # Verify task was queued
        queue_length = await test_redis_connection.llen("v10r:tasks")
        assert queue_length == 1
        
        # Retrieve and verify task
        task_data = await test_redis_connection.rpop("v10r:tasks")
        retrieved_task = json.loads(task_data)
        
        assert retrieved_task["table"] == "articles"
        assert retrieved_task["text"] == "This is content to be vectorized"
    
    async def test_priority_queue_handling(self, test_redis_connection, clean_test_data):
        """Test priority-based task handling."""
        # Queue high and low priority tasks
        high_priority_task = {"priority": "high", "task_id": "urgent_1"}
        low_priority_task = {"priority": "low", "task_id": "batch_1"}
        
        # Queue to different priority channels
        await test_redis_connection.lpush("v10r:tasks:high", json.dumps(high_priority_task))
        await test_redis_connection.lpush("v10r:tasks:low", json.dumps(low_priority_task))
        
        # Verify tasks are in appropriate queues
        high_count = await test_redis_connection.llen("v10r:tasks:high")
        low_count = await test_redis_connection.llen("v10r:tasks:low")
        
        assert high_count == 1
        assert low_count == 1


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPipeline:
    """End-to-end pipeline tests (marked as slow)."""
    
    async def test_complete_vectorization_flow(
        self, 
        test_db_connection, 
        test_redis_connection,
        mock_infinity_url,
        clean_test_data
    ):
        """Test complete flow from database insert to vector storage."""
        # 1. Insert test data
        await test_db_connection.execute("""
            INSERT INTO articles (id, title, content, author) 
            VALUES (999, 'E2E Test Article', 'Content for end-to-end testing', 'Test Author')
        """)
        
        # 2. Simulate vectorization request
        vectorization_task = {
            "database": "v10r_test_db",
            "schema": "public", 
            "table": "articles",
            "row_id": 999,
            "text_column": "content",
            "vector_column": "content_vector",
            "model_column": "content_embedding_model",
            "text": "Content for end-to-end testing"
        }
        
        await test_redis_connection.lpush("v10r:tasks", json.dumps(vectorization_task))
        
        # 3. Simulate worker processing (call mock API)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                mock_infinity_url,
                json={
                    "model": "BAAI/bge-m3",
                    "input": [vectorization_task["text"]]
                }
            )
            
            assert response.status_code == 200
            embedding_data = response.json()
            embedding = embedding_data["data"][0]["embedding"]
            
            # Truncate to 768 dimensions to match content_vector column
            embedding = embedding[:768]
        
        # 4. Simulate updating database with vector
        # Convert embedding list to pgvector string format
        vector_str = f"[{','.join(map(str, embedding))}]"
        
        await test_db_connection.execute("""
            UPDATE articles 
            SET content_vector = $1::vector, 
                content_embedding_model = $2,
                content_vector_created_at = NOW()
            WHERE id = $3
        """, vector_str, "BAAI/bge-m3", 999)
        
        # 5. Verify end-to-end result
        result = await test_db_connection.fetchrow("""
            SELECT content, content_vector::text as content_vector, content_embedding_model, content_vector_created_at
            FROM articles 
            WHERE id = 999
        """)
        
        assert result['content'] == "Content for end-to-end testing"
        assert result['content_vector'] is not None
        # pgvector returns vectors as strings, parse it
        vector_str = result['content_vector']
        vector_values = [float(x) for x in vector_str.strip('[]').split(',')]
        assert len(vector_values) == 768
        assert result['content_embedding_model'] == "BAAI/bge-m3"
        assert result['content_vector_created_at'] is not None


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling in integration environment."""
    
    async def test_database_connection_resilience(self):
        """Test handling of database connection issues."""
        # Test with invalid connection parameters
        with pytest.raises(Exception):  # Should be more specific exception
            await asyncpg.connect(
                host='localhost',
                port=15432,
                user='invalid_user',
                password='wrong_password',
                database='v10r_test_db'
            )
    
    async def test_redis_connection_resilience(self):
        """Test handling of Redis connection issues."""
        # Test with invalid Redis connection
        import redis.asyncio as redis
        
        invalid_client = redis.Redis(
            host='localhost',
            port=99999,  # Invalid port
            decode_responses=True
        )
        
        with pytest.raises(Exception):
            await invalid_client.ping()
        
        await invalid_client.aclose()
    
    async def test_api_error_handling(self):
        """Test handling of API errors."""
        async with httpx.AsyncClient() as client:
            # Test invalid endpoint
            with pytest.raises(httpx.ConnectError):
                await client.post(
                    "http://localhost:99999/v1/embeddings",  # Invalid port
                    json={"model": "test", "input": ["test"]},
                    timeout=5
                )


# ============================================================================
# Test Configuration and Utilities
# ============================================================================

@pytest.mark.integration
def test_pytest_markers_work():
    """Test that pytest markers are working correctly."""
    # This test verifies the test environment is set up correctly
    assert True


if __name__ == "__main__":
    # Allow running tests directly, but recommend using pytest
    print("Run these tests with: pytest tests/integration/ -m integration")
    print("Or for all tests: pytest tests/integration/") 