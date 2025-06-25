"""
Pytest configuration and shared fixtures for v10r tests.

This module provides shared fixtures and utilities for testing all v10r components.
"""

import asyncio
import json
import os
import tempfile
from typing import Dict, List, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock
import pytest
import yaml
from aioresponses import aioresponses
from aioresponses import CallbackResult

from v10r.config import V10rConfig, EmbeddingConfig, DatabaseConfig, DatabaseConnection
from v10r.embedding import EmbeddingClientFactory


# ============================================================================
# Test Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding_configs() -> Dict[str, Dict[str, Any]]:
    """Sample embedding configurations for testing."""
    return {
        "infinity_bge_m3": {
            "provider": "infinity",
            "endpoint": "http://localhost:7997",
            "model": "BAAI/bge-m3",
            "dimensions": 1024,
            "batch_size": 30,
            "timeout": 60,
            "max_retries": 3,
            "retry_delay": 2.0,
        },
        "local_infinity": {
            "provider": "infinity",
            "endpoint": "http://localhost:8080",
            "model": "BAAI/bge-m3",
            "dimensions": 1024,
            "batch_size": 20,
            "timeout": 30,
            "max_retries": 2,
            "retry_delay": 1.0,
        },
        "openai_test": {
            "provider": "openai",
            "api_key": "test-api-key",
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "batch_size": 100,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
        "custom_test": {
            "provider": "custom",
            "endpoint": "http://localhost:9090",
            "api_key": "custom-api-key",
            "model": "custom-model",
            "dimensions": 512,
            "batch_size": 20,
            "timeout": 30,
            "max_retries": 3,
            "retry_delay": 1.0,
        },
    }


@pytest.fixture
def embedding_config_openai(sample_embedding_configs) -> EmbeddingConfig:
    """OpenAI embedding configuration for testing."""
    return EmbeddingConfig(**sample_embedding_configs["openai_test"])


@pytest.fixture
def embedding_config_infinity_bge_m3(sample_embedding_configs) -> EmbeddingConfig:
    """Infinity BGE-M3 embedding configuration for testing."""
    return EmbeddingConfig(**sample_embedding_configs["infinity_bge_m3"])


@pytest.fixture
def embedding_config_infinity_local(sample_embedding_configs) -> EmbeddingConfig:
    """Local Infinity embedding configuration for testing."""
    return EmbeddingConfig(**sample_embedding_configs["local_infinity"])


@pytest.fixture
def embedding_config_custom(sample_embedding_configs) -> EmbeddingConfig:
    """Custom API embedding configuration for testing."""
    return EmbeddingConfig(**sample_embedding_configs["custom_test"])


@pytest.fixture
def sample_v10r_config(sample_embedding_configs) -> V10rConfig:
    """Complete v10r configuration for testing."""
    config_data = {
        "service_name": "v10r-test",
        "debug": True,
        "databases": [
            {
                "name": "test_db",
                "connection": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db",
                    "user": "test_user",
                    "password": "test_password",
                },
                "tables": [
                    {
                        "schema": "public",
                        "table": "test_documents",
                        "text_column": "content",
                        "id_column": "id",
                        "vector_column": "content_vector",
                        "model_column": "content_embedding_model",
                        "embedding_config": "infinity_bge_m3",
                    }
                ],
            }
        ],
        "embeddings": sample_embedding_configs,
        "queue": {
            "type": "redis",
            "connection": {"host": "localhost", "port": 6379, "db": 0},
        },
        "workers": {"concurrency": 2},
    }
    return V10rConfig(**config_data)


@pytest.fixture
def temp_config_file(sample_v10r_config) -> str:
    """Temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_v10r_config.dict(), f)
        return f.name


# ============================================================================
# Mock API Fixtures
# ============================================================================

@pytest.fixture
def mock_embedding_response() -> Dict[str, Any]:
    """Sample embedding API response."""
    return {
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.1] * 1024,  # BGE-M3 dimensions
            },
            {
                "object": "embedding", 
                "index": 1,
                "embedding": [0.2] * 1024,
            },
        ],
        "model": "BAAI/bge-m3",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


@pytest.fixture
def mock_openai_responses():
    """Mock OpenAI API responses using aioresponses."""
    with aioresponses() as m:
        def dynamic_response(url, **kwargs):
            # Extract texts from request payload
            json_data = kwargs.get('json', {})
            texts = json_data.get('input', [])
            if isinstance(texts, str):
                texts = [texts]
            
            # Return embeddings for each text
            response_data = {
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 1536}
                    for i in range(len(texts))
                ],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": len(texts) * 5, "total_tokens": len(texts) * 5},
            }
            return CallbackResult(payload=response_data)
        
        m.post(
            "https://api.openai.com/v1/embeddings",
            callback=dynamic_response,
        )
        yield m


@pytest.fixture  
def mock_infinity_responses():
    """Mock Infinity API responses for both Fuzu and local deployments."""
    with aioresponses() as m:
        def dynamic_infinity_response(url, **kwargs):
            # Extract texts from request payload
            json_data = kwargs.get('json', {})
            texts = json_data.get('input', [])
            if isinstance(texts, str):
                texts = [texts]
            
            # Return embeddings for each text
            response_data = {
                "data": [
                    {"object": "embedding", "index": i, "embedding": [0.1] * 1024}
                    for i in range(len(texts))
                ],
                "model": "BAAI/bge-m3",
            }
            return CallbackResult(payload=response_data)
        
        # Infinity endpoint
        m.post(
            "http://localhost:7997/v1/embeddings",
            callback=dynamic_infinity_response,
        )
        # Local development endpoint
        m.post(
            "http://localhost:8080/v1/embeddings",
            callback=dynamic_infinity_response,
        )
        # Models endpoints
        m.get(
            "http://localhost:7997/v1/models",
            payload={
                "data": [
                    {
                        "id": "BAAI/bge-m3",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "infinity",
                    }
                ]
            },
        )
        m.get(
            "http://localhost:8080/v1/models",
            payload={
                "data": [
                    {
                        "id": "BAAI/bge-m3",
                        "object": "model",
                        "created": 1234567890,
                        "owned_by": "infinity",
                    }
                ]
            },
        )
        # Health endpoints
        m.get(
            "http://localhost:7997/health",
            payload={"status": "healthy", "models": ["BAAI/bge-m3"]}
        )
        m.get(
            "http://localhost:8080/health",
            payload={"status": "healthy", "models": ["BAAI/bge-m3"]}
        )
        yield m


@pytest.fixture
def mock_custom_responses():
    """Mock custom API responses."""
    with aioresponses() as m:
        def dynamic_custom_response(url, **kwargs):
            # Extract texts from request payload
            json_data = kwargs.get('json', {})
            texts = json_data.get('input', [])
            if isinstance(texts, str):
                texts = [texts]
            
            # Return embeddings for each text
            response_data = {
                "data": [
                    {"embedding": [0.1] * 512, "index": i}
                    for i in range(len(texts))
                ]
            }
            return CallbackResult(payload=response_data)
        
        # Embeddings endpoint - supports various response formats
        m.post(
            "http://localhost:9090/v1/embeddings",
            callback=dynamic_custom_response,
        )
        # Models endpoint
        m.get(
            "http://localhost:9090/v1/models",
            payload={
                "data": [
                    {"id": "custom-model", "object": "model", "owned_by": "custom"}
                ]
            },
        )
        yield m


# ============================================================================
# Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_texts() -> List[str]:
    """Sample texts for embedding testing."""
    return [
        "This is a test document.",
        "Another example text for embedding.",
        "Short text",
        "A longer piece of text that contains multiple sentences. This helps test how the embedding service handles various text lengths and complexities.",
        "",  # Empty string edge case
        "   ",  # Whitespace only
        "Text with special characters: àáâãäåæçèéêëìíîïðñòóôõö",
        "Multi-line\ntext\nwith\nbreaks",
    ]


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """Sample embedding vectors for testing (BGE-M3 dimensions)."""
    return [
        [0.1, 0.2, 0.3] * 341 + [0.1],  # 1024 dimensions
        [0.4, 0.5, 0.6] * 341 + [0.4],
        [0.7, 0.8, 0.9] * 341 + [0.7],
    ]


# ============================================================================
# Database Test Fixtures
# ============================================================================

@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    conn = AsyncMock()
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.fetchrow = AsyncMock()
    conn.fetchval = AsyncMock()
    return conn


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.get = AsyncMock()
    redis_mock.set = AsyncMock()
    redis_mock.delete = AsyncMock()
    redis_mock.lpush = AsyncMock()
    redis_mock.rpop = AsyncMock()
    return redis_mock


# ============================================================================
# Test Utilities
# ============================================================================

class MockEmbeddingResponse:
    """Utility class for creating mock embedding responses."""
    
    @staticmethod
    def openai_format(embeddings: List[List[float]], model: str = "BAAI/bge-m3") -> Dict[str, Any]:
        """Create OpenAI-format response."""
        return {
            "data": [
                {"object": "embedding", "index": i, "embedding": emb}
                for i, emb in enumerate(embeddings)
            ],
            "model": model,
            "usage": {"prompt_tokens": len(embeddings) * 5, "total_tokens": len(embeddings) * 5},
        }
    
    @staticmethod
    def direct_format(embeddings: List[List[float]]) -> Dict[str, Any]:
        """Create direct embeddings format response."""
        return {"embeddings": embeddings}
    
    @staticmethod
    def simple_array(embeddings: List[List[float]]) -> List[List[float]]:
        """Return embeddings as simple array."""
        return embeddings


@pytest.fixture
def mock_response_generator():
    """Fixture providing MockEmbeddingResponse utility."""
    return MockEmbeddingResponse


# ============================================================================
# Error Testing Fixtures  
# ============================================================================

@pytest.fixture
def api_error_responses():
    """Various API error responses for testing error handling."""
    return {
        "rate_limit": {
            "status": 429,
            "headers": {"retry-after": "60"},
            "payload": {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        },
        "auth_error": {
            "status": 401,
            "payload": {
                "error": {
                    "message": "Invalid API key",
                    "type": "invalid_request_error",
                    "code": "invalid_api_key",
                }
            },
        },
        "validation_error": {
            "status": 422,
            "payload": {
                "detail": "Model not found"
            },
        },
        "server_error": {
            "status": 500,
            "payload": {
                "error": {
                    "message": "Internal server error",
                    "type": "api_error",
                }
            },
        },
    }


# ============================================================================
# Performance Testing Fixtures
# ============================================================================

@pytest.fixture
def large_text_dataset() -> List[str]:
    """Large dataset for performance testing."""
    return [f"Test document number {i} with some content." for i in range(1000)]


@pytest.fixture
def performance_config() -> EmbeddingConfig:
    """Configuration optimized for performance testing."""
    return EmbeddingConfig(
        provider="infinity",
        endpoint="http://localhost:7997",
        model="BAAI/bge-m3",
        dimensions=1024,
        batch_size=50,   # Optimized batch size for BGE-M3
        timeout=120,     # Longer timeout for large batches
        max_retries=1,   # Fewer retries for faster tests
        retry_delay=0.1, # Shorter delay
    )


# ============================================================================
# Integration Test Fixtures
# ============================================================================

@pytest.fixture
async def embedding_client_factory():
    """Factory for creating embedding clients in tests."""
    clients = []
    
    def create_client(config: EmbeddingConfig):
        client = EmbeddingClientFactory.create_client(config)
        clients.append(client)
        return client
    
    yield create_client
    
    # Cleanup - close all created clients
    for client in clients:
        if hasattr(client, 'close'):
            await client.close()


# ============================================================================
# Environment Setup
# ============================================================================

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Backup original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_env = {
        "V10R_DEBUG": "true",
        "V10R_LOG_LEVEL": "DEBUG",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_DB": "test_db",
        "POSTGRES_USER": "test_user", 
        "POSTGRES_PASSWORD": "test_password",
        "REDIS_HOST": "localhost",
        "OPENAI_API_KEY": "test-openai-key",
        "INFINITY_ENDPOINT": "http://localhost:7997",  # Production endpoint
        "INFINITY_ENDPOINT_LOCAL": "http://localhost:8080",  # Local dev
        "CUSTOM_EMBEDDING_ENDPOINT": "http://localhost:9090",
        "CUSTOM_API_KEY": "test-custom-key",
    }
    
    os.environ.update(test_env)
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Async Test Utilities
# ============================================================================

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Markers Helpers
# ============================================================================

def requires_network(func):
    """Mark test as requiring network access."""
    return pytest.mark.network(func)


def requires_database(func):
    """Mark test as requiring database."""
    return pytest.mark.database(func)


def requires_redis(func):
    """Mark test as requiring Redis."""
    return pytest.mark.redis(func)


def slow_test(func):
    """Mark test as slow (> 1 second)."""
    return pytest.mark.slow(func)


# Make markers available as fixtures
@pytest.fixture
def test_markers():
    """Test marker utilities."""
    return {
        "network": requires_network,
        "database": requires_database,
        "redis": requires_redis,
        "slow": slow_test,
    }


# Docker Compose Integration Test Fixtures
@pytest.fixture(scope="session")
def docker_compose_services():
    """
    Ensure Docker Compose services are available for integration testing.
    In development, services are expected to already be running.
    """
    import subprocess
    import time
    import socket
    
    # Function to check if a port is open
    def is_port_open(host, port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                return sock.connect_ex((host, port)) == 0
        except:
            return False
    
    # Check if services are already running
    postgres_ready = is_port_open('localhost', 15432)
    redis_ready = is_port_open('localhost', 16379)
    infinity_ready = is_port_open('localhost', 18080)
    
    if not all([postgres_ready, redis_ready, infinity_ready]):
        # Try to start services if they're not running
        try:
            subprocess.run([
                "docker-compose", "-f", "docker-compose.test.yml", 
                "up", "-d", "--build"
            ], check=True, capture_output=True)
            
            # Wait for services to become ready
            for _ in range(60):  # 60 second timeout
                if all([
                    is_port_open('localhost', 15432),
                    is_port_open('localhost', 16379), 
                    is_port_open('localhost', 18080)
                ]):
                    break
                time.sleep(1)
            else:
                raise Exception("Services did not become ready in time")
        except subprocess.CalledProcessError as e:
            # Services might already be running, continue if ports are accessible
            if not all([is_port_open('localhost', 15432), is_port_open('localhost', 16379)]):
                raise Exception(f"Docker Compose failed and services not accessible: {e}")
    
    # Test PostgreSQL connection
    import asyncpg
    import asyncio
    for _ in range(10):
        try:
            conn = asyncio.get_event_loop().run_until_complete(
                asyncpg.connect(
                    host='localhost',
                    port=15432,
                    user='v10r_test',
                    password='test_password_123',
                    database='v10r_test_db'
                )
            )
            asyncio.get_event_loop().run_until_complete(conn.close())
            break
        except:
            time.sleep(1)
    else:
        raise Exception("PostgreSQL connection test failed")
    
    # Test Redis connection
    import redis
    for _ in range(10):
        try:
            client = redis.Redis(host='localhost', port=16379, db=0)
            client.ping()
            client.close()
            break
        except:
            time.sleep(1)
    else:
        raise Exception("Redis connection test failed")
    
    yield
    
    # Note: We don't cleanup automatically since services might be shared
    # Run `docker-compose -f docker-compose.test.yml down -v` manually if needed

@pytest.fixture
async def test_db_connection(docker_compose_services):
    """
    Provide a test database connection for integration tests.
    """
    import asyncpg
    
    conn = await asyncpg.connect(
        host='localhost',
        port=15432,
        user='v10r_test',
        password='test_password_123',
        database='v10r_test_db'
    )
    
    yield conn
    
    await conn.close()

@pytest.fixture
async def test_redis_connection(docker_compose_services):
    """
    Provide a test Redis connection for integration tests.
    """
    import redis.asyncio as redis
    
    client = redis.Redis(
        host='localhost',
        port=16379,
        db=0,
        decode_responses=True
    )
    
    # Test connection
    await client.ping()
    
    yield client
    
    await client.aclose()

@pytest.fixture
def mock_infinity_url(docker_compose_services):
    """
    Provide the mock Infinity server URL for testing.
    """
    return "http://localhost:18080/v1/embeddings"

@pytest.fixture
async def clean_test_data(test_db_connection, test_redis_connection):
    """
    Clean test data before and after each test.
    """
    # Clean before test
    await test_db_connection.execute("TRUNCATE articles, products, user_profiles CASCADE")
    # Clean Redis v10r keys
    async for key in test_redis_connection.scan_iter("v10r:*"):
        await test_redis_connection.delete(key)
    
    yield
    
    # Clean after test
    await test_db_connection.execute("TRUNCATE articles, products, user_profiles CASCADE")
    # Clean Redis v10r keys
    async for key in test_redis_connection.scan_iter("v10r:*"):
        await test_redis_connection.delete(key)

# Mark integration tests properly
@pytest.fixture(autouse=True)
def mark_integration_tests(request):
    """
    Automatically mark tests in integration directory.
    """
    if "integration" in str(request.fspath):
        request.node.add_marker(pytest.mark.integration) 