"""
Tests for v10r.embedding.base module.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from v10r.embedding.base import EmbeddingClient
from v10r.config import EmbeddingConfig
from v10r.exceptions import (
    EmbeddingError, 
    EmbeddingAPIError, 
    ValidationError, 
    RateLimitError, 
    APITimeoutError
)


class ConcreteEmbeddingClient(EmbeddingClient):
    """Concrete implementation for testing abstract base class."""
    
    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self.embed_texts_called = []
        self.embed_single_called = []
        self.health_check_called = []
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Mock implementation of embed_texts."""
        self.embed_texts_called.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    async def embed_single(self, text: str) -> list[float]:
        """Mock implementation of embed_single."""
        self.embed_single_called.append(text)
        return [0.1, 0.2, 0.3]
    
    async def health_check(self) -> dict[str, any]:
        """Mock implementation of health_check."""
        self.health_check_called.append(True)
        return {"status": "healthy", "model": self.model}


class TestEmbeddingClient:
    """Test EmbeddingClient abstract base class."""
    
    @pytest.fixture
    def valid_config(self):
        """Valid embedding configuration for testing."""
        return EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=768,
            batch_size=10,
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
            api_key="test-key"
        )
    
    @pytest.fixture
    def client(self, valid_config):
        """Create concrete embedding client for testing."""
        return ConcreteEmbeddingClient(valid_config)
    
    def test_init_with_valid_config(self, valid_config):
        """Test initialization with valid configuration."""
        client = ConcreteEmbeddingClient(valid_config)
        
        assert client.config == valid_config
        assert client.model == "test-model"
        assert client.dimensions == 768
        assert client.batch_size == 10
    
    def test_init_validates_config(self):
        """Test initialization validates configuration."""
        # Test invalid dimensions
        config = EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=0,  # Invalid
            batch_size=10,
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Dimensions must be positive"):
            ConcreteEmbeddingClient(config)
    
    def test_validate_config_dimensions(self):
        """Test configuration validation for dimensions."""
        config = EmbeddingConfig(
            provider="custom",
            model="test-model", 
            dimensions=-1,
            batch_size=10,
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Dimensions must be positive"):
            ConcreteEmbeddingClient(config)
    
    def test_validate_config_batch_size(self):
        """Test configuration validation for batch size."""
        config = EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=768,
            batch_size=0,  # Invalid
            timeout=30.0,
            max_retries=3,
            retry_delay=1.0,
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Batch size must be positive"):
            ConcreteEmbeddingClient(config)
    
    def test_validate_config_timeout(self):
        """Test configuration validation for timeout."""
        config = EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=768,
            batch_size=10,
            timeout=0,  # Invalid
            max_retries=3,
            retry_delay=1.0,
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Timeout must be positive"):
            ConcreteEmbeddingClient(config)
    
    def test_validate_config_max_retries(self):
        """Test configuration validation for max retries."""
        config = EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=768,
            batch_size=10,
            timeout=30.0,
            max_retries=-1,  # Invalid
            retry_delay=1.0,
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Max retries cannot be negative"):
            ConcreteEmbeddingClient(config)
    
    def test_validate_config_retry_delay(self):
        """Test configuration validation for retry delay."""
        config = EmbeddingConfig(
            provider="custom",
            model="test-model",
            dimensions=768,
            batch_size=10,
            timeout=30.0,
            max_retries=3,
            retry_delay=0,  # Invalid
            api_key="test-key"
        )
        
        with pytest.raises(ValidationError, match="Retry delay must be positive"):
            ConcreteEmbeddingClient(config)
    
    def test_properties(self, client):
        """Test property getters."""
        assert client.model == "test-model"
        assert client.dimensions == 768
        assert client.batch_size == 10
    
    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self, client):
        """Test embed_batch with empty list."""
        result = await client.embed_batch([])
        assert result == []
        assert len(client.embed_texts_called) == 0
    
    @pytest.mark.asyncio
    async def test_embed_batch_single_batch(self, client):
        """Test embed_batch with texts that fit in single batch."""
        texts = ["hello", "world"]
        
        result = await client.embed_batch(texts)
        
        assert len(result) == 2
        assert len(client.embed_texts_called) == 1
        assert client.embed_texts_called[0] == texts
    
    @pytest.mark.asyncio
    async def test_embed_batch_multiple_batches(self, client):
        """Test embed_batch with texts requiring multiple batches."""
        # Set small batch size for testing
        client.config.batch_size = 2
        texts = ["one", "two", "three", "four", "five"]
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await client.embed_batch(texts)
        
        assert len(result) == 5
        assert len(client.embed_texts_called) == 3  # 3 batches: [2, 2, 1]
        
        # Check batches were called correctly
        assert client.embed_texts_called[0] == ["one", "two"]
        assert client.embed_texts_called[1] == ["three", "four"]
        assert client.embed_texts_called[2] == ["five"]
        
        # Check sleep was called between batches
        assert mock_sleep.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_embed_batch_batch_size_one(self, client):
        """Test embed_batch with batch size of 1."""
        client.config.batch_size = 1
        texts = ["hello", "world"]
        
        result = await client.embed_batch(texts)
        
        assert len(result) == 2
        # Should call embed_texts directly, not split into batches
        assert len(client.embed_texts_called) == 1
        assert client.embed_texts_called[0] == texts
    
    def test_validate_texts_empty_list(self, client):
        """Test _validate_texts with empty list."""
        with pytest.raises(ValidationError, match="Cannot embed empty list of texts"):
            client._validate_texts([])
    
    def test_validate_texts_non_string(self, client):
        """Test _validate_texts with non-string input."""
        with pytest.raises(ValidationError, match="Text at index 1 is not a string"):
            client._validate_texts(["hello", 123, "world"])
    
    def test_validate_texts_empty_string(self, client):
        """Test _validate_texts with empty string."""
        # Should not raise exception but log warning
        with patch.object(client.logger, 'warning') as mock_warn:
            client._validate_texts(["hello", "", "world"])
            mock_warn.assert_called_once()
    
    def test_validate_texts_whitespace_only(self, client):
        """Test _validate_texts with whitespace-only string."""
        with patch.object(client.logger, 'warning') as mock_warn:
            client._validate_texts(["hello", "   ", "world"])
            mock_warn.assert_called_once()
    
    def test_validate_texts_valid(self, client):
        """Test _validate_texts with valid input."""
        # Should not raise exception
        client._validate_texts(["hello", "world", "test"])
    
    def test_validate_embedding_response_correct_count(self, client):
        """Test _validate_embedding_response with correct count."""
        embeddings = [[0.1] * 768, [0.3] * 768, [0.5] * 768]
        
        # Should not raise exception
        client._validate_embedding_response(embeddings, 3)
    
    def test_validate_embedding_response_wrong_count(self, client):
        """Test _validate_embedding_response with wrong count."""
        embeddings = [[0.1] * 768, [0.3] * 768]
        
        with pytest.raises(ValidationError, match="Expected 3 embeddings, got 2"):
            client._validate_embedding_response(embeddings, 3)
    
    def test_validate_embedding_response_wrong_dimensions(self, client):
        """Test _validate_embedding_response with wrong dimensions."""
        embeddings = [
            [0.1] * 768,    # Correct dimensions
            [0.4, 0.5]      # 2 dimensions - wrong!
        ]
        
        with pytest.raises(ValidationError, match="has 2 dimensions, expected 768"):
            client._validate_embedding_response(embeddings, 2)
    
    def test_validate_embedding_response_non_numeric(self, client):
        """Test _validate_embedding_response with non-numeric values."""
        embeddings = [
            [0.1, 0.2, "invalid"] + [0.0] * 765,  # String instead of float, padded to 768
            [0.4] * 768
        ]
        
        with pytest.raises(ValidationError, match="contains non-numeric values"):
            client._validate_embedding_response(embeddings, 2)
    
    @pytest.mark.asyncio
    async def test_context_manager_aenter(self, client):
        """Test async context manager __aenter__."""
        result = await client.__aenter__()
        assert result is client
    
    @pytest.mark.asyncio
    async def test_context_manager_aexit(self, client):
        """Test async context manager __aexit__."""
        with patch.object(client, 'close', new_callable=AsyncMock) as mock_close:
            await client.__aexit__(None, None, None)
            mock_close.assert_called_once()
    
    @pytest.mark.asyncio  
    async def test_close(self, client):
        """Test close method."""
        # Base implementation should not raise exception
        await client.close()
    
    def test_repr(self, client):
        """Test __repr__ method."""
        repr_str = repr(client)
        assert "ConcreteEmbeddingClient" in repr_str
        assert "test-model" in repr_str
        assert "768" in repr_str
    
    def test_handle_api_error_rate_limit(self, client):
        """Test _handle_api_error with rate limit error."""
        original_error = Exception("Rate limit exceeded")
        
        result = client._handle_api_error(original_error, "embed_texts")
        
        assert isinstance(result, EmbeddingAPIError)
        assert "Rate limit exceeded" in str(result)
    
    def test_handle_api_error_timeout(self, client):
        """Test _handle_api_error with timeout error."""
        original_error = Exception("Request timeout")
        
        result = client._handle_api_error(original_error, "embed_texts")
        
        assert isinstance(result, EmbeddingAPIError)
        assert "Request timeout" in str(result)
    
    def test_handle_api_error_generic(self, client):
        """Test _handle_api_error with generic error."""
        original_error = Exception("Something went wrong")
        
        result = client._handle_api_error(original_error, "embed_texts")
        
        assert isinstance(result, EmbeddingAPIError)
        assert "Something went wrong" in str(result)
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_first_try(self, client):
        """Test _retry_with_backoff succeeds on first try."""
        mock_func = AsyncMock(return_value="success")
        
        result = await client._retry_with_backoff(mock_func, "arg1", kwarg1="value1")
        
        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", kwarg1="value1")
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_success_after_retries(self, client):
        """Test _retry_with_backoff succeeds after retries."""
        mock_func = AsyncMock(side_effect=[
            Exception("Temporary error"),
            Exception("Another error"),
            "success"
        ])
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            result = await client._retry_with_backoff(mock_func)
        
        assert result == "success"
        assert mock_func.call_count == 3
        assert mock_sleep.call_count == 2  # Slept between retries
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_max_retries_exceeded(self, client):
        """Test _retry_with_backoff when max retries exceeded."""
        mock_func = AsyncMock(side_effect=Exception("Persistent error"))
        
        with patch('asyncio.sleep', new_callable=AsyncMock):
            with pytest.raises(EmbeddingAPIError):
                await client._retry_with_backoff(mock_func)
        
        # Should try initial + max_retries times
        assert mock_func.call_count == client.config.max_retries + 1
    
    @pytest.mark.asyncio
    async def test_retry_with_backoff_exponential_backoff(self, client):
        """Test _retry_with_backoff uses exponential backoff."""
        mock_func = AsyncMock(side_effect=[
            Exception("Error 1"),
            Exception("Error 2"), 
            Exception("Error 3"),
            "success"
        ])
        
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await client._retry_with_backoff(mock_func)
        
        # Check exponential backoff delays
        expected_delays = [
            client.config.retry_delay,      # 1.0
            client.config.retry_delay * 2,  # 2.0
            client.config.retry_delay * 4   # 4.0
        ]
        
        actual_delays = [call[0][0] for call in mock_sleep.call_args_list]
        assert actual_delays == expected_delays


class FailingEmbeddingClient(EmbeddingClient):
    """Embedding client that raises exceptions for error testing."""
    
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise Exception("API Error")
    
    async def embed_single(self, text: str) -> list[float]:
        raise Exception("API Error")
    
    async def health_check(self) -> dict[str, any]:
        raise Exception("Service Unavailable")


class TestEmbeddingClientErrorHandling:
    """Test error handling in EmbeddingClient."""
    
    @pytest.fixture
    def valid_config(self):
        """Create valid embedding configuration."""
        return EmbeddingConfig(
            provider="custom",
            model="test-model", 
            dimensions=768,
            batch_size=10,
            timeout=30.0,
            max_retries=1,  # Low retries for faster tests
            retry_delay=0.1,
            api_key="test-key"
        )
    
    @pytest.fixture
    def failing_client(self, valid_config):
        """Create failing embedding client for error testing."""
        return FailingEmbeddingClient(valid_config)
    
    @pytest.mark.asyncio
    async def test_embed_batch_with_validation_error(self, failing_client):
        """Test embed_batch with validation error."""
        # Invalid input should be caught before calling embed_texts
        with pytest.raises(ValidationError):
            await failing_client.embed_batch([123, "text"])
    
    @pytest.mark.asyncio
    async def test_embed_batch_handles_embed_texts_error(self, failing_client):
        """Test embed_batch properly handles embed_texts errors."""
        # Valid input but embed_texts fails
        with pytest.raises(Exception, match="API Error"):
            await failing_client.embed_batch(["valid text"])


class TestEmbeddingClientIntegration:
    """Integration tests for EmbeddingClient functionality."""
    
    @pytest.fixture
    def config_with_large_batch(self):
        """Config with large batch size for integration testing."""
        return EmbeddingConfig(
            provider="custom",
            model="large-model",
            dimensions=1536,
            batch_size=100,
            timeout=60.0,
            max_retries=5,
            retry_delay=2.0,
            api_key="integration-key"
        )
    
    @pytest.mark.asyncio
    async def test_full_embedding_workflow(self, config_with_large_batch):
        """Test complete embedding workflow."""
        client = ConcreteEmbeddingClient(config_with_large_batch)
        
        # Test with various input sizes
        test_cases = [
            [],  # Empty
            ["single text"],  # Single item
            ["text1", "text2", "text3"],  # Small batch
            [f"text{i}" for i in range(150)]  # Large batch requiring splitting
        ]
        
        for texts in test_cases:
            if not texts:
                result = await client.embed_batch(texts)
                assert result == []
            else:
                result = await client.embed_batch(texts)
                assert len(result) == len(texts)
                assert all(len(embedding) == 3 for embedding in result)  # Mock returns 3D vectors
    
    @pytest.mark.asyncio
    async def test_context_manager_usage(self, config_with_large_batch):
        """Test using client as async context manager."""
        async with ConcreteEmbeddingClient(config_with_large_batch) as client:
            result = await client.embed_batch(["test"])
            assert len(result) == 1
            
        # Client should be properly closed after context exit
        # (In real implementation, this might close connections, etc.) 