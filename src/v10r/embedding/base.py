"""
Abstract base class for embedding clients.

This module provides the base interface that all embedding clients must implement.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import logging

from ..config import EmbeddingConfig
from ..exceptions import EmbeddingError, EmbeddingAPIError, ValidationError, RateLimitError, APITimeoutError


logger = logging.getLogger(__name__)


class EmbeddingClient(ABC):
    """
    Abstract base class for all embedding clients.
    
    This class defines the interface that all embedding providers must implement.
    It provides common functionality like configuration validation, error handling,
    and async context management.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize the embedding client with configuration.
        
        Args:
            config: Embedding configuration containing provider settings
            
        Raises:
            ValidationError: If configuration is invalid
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self) -> None:
        """
        Validate the embedding configuration.
        
        Raises:
            ValidationError: If configuration is invalid
        """
        if self.config.dimensions <= 0:
            raise ValidationError("Dimensions must be positive")
            
        if self.config.batch_size <= 0:
            raise ValidationError("Batch size must be positive")
            
        if self.config.timeout <= 0:
            raise ValidationError("Timeout must be positive")
            
        if self.config.max_retries < 0:
            raise ValidationError("Max retries cannot be negative")
            
        if self.config.retry_delay <= 0:
            raise ValidationError("Retry delay must be positive")
    
    @property
    def model(self) -> str:
        """Get the model name."""
        return self.config.model

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.config.dimensions

    @property
    def batch_size(self) -> int:
        """Get the batch size."""
        return self.config.batch_size

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors, one per input text
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If input validation fails
        """
        pass
    
    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            EmbeddingError: If embedding generation fails
            ValidationError: If input validation fails
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the embedding service.
        
        Returns:
            Dictionary with health status information
            
        Raises:
            APIError: If health check fails
        """
        pass
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts in batches according to the configured batch size.
        
        This method handles batching automatically and can be overridden
        by specific clients for optimized batch processing.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
            
        # Validate input texts
        self._validate_texts(texts)
        
        # If batch size is 1 or we have fewer texts than batch size, process directly
        if self.config.batch_size == 1 or len(texts) <= self.config.batch_size:
            return await self.embed_texts(texts)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = await self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to be respectful to API limits
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)
                
        return all_embeddings
    
    def _validate_texts(self, texts: List[str]) -> None:
        """
        Validate input texts.
        
        Args:
            texts: List of texts to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not texts:
            raise ValidationError("Cannot embed empty list of texts")
            
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(f"Text at index {i} is not a string: {type(text)}")
                
            if not text.strip():
                self.logger.warning(f"Empty or whitespace-only text at index {i}")
    
    def _validate_embedding_response(self, embeddings: List[List[float]], 
                                   expected_count: int) -> None:
        """
        Validate embedding response from API.
        
        Args:
            embeddings: List of embedding vectors
            expected_count: Expected number of embeddings
            
        Raises:
            ValidationError: If validation fails
        """
        if len(embeddings) != expected_count:
            raise ValidationError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}"
            )
            
        for i, embedding in enumerate(embeddings):
            if not isinstance(embedding, list):
                raise ValidationError(f"Embedding {i} is not a list: {type(embedding)}")
                
            if len(embedding) != self.config.dimensions:
                raise ValidationError(
                    f"Embedding {i} has {len(embedding)} dimensions, "
                    f"expected {self.config.dimensions}"
                )
                
            if not all(isinstance(x, (int, float)) for x in embedding):
                raise ValidationError(f"Embedding {i} contains non-numeric values")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def close(self) -> None:
        """
        Close the client and clean up resources.
        
        This method can be overridden by specific clients that need
        to close connections, sessions, etc.
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (
            f"{self.__class__.__name__}("
            f"provider={self.config.provider}, "
            f"model={self.config.model}, "
            f"dimensions={self.config.dimensions})"
        ) 

    def _handle_api_error(self, error: Exception, operation: str) -> Exception:
        """
        Convert API errors to appropriate embedding exceptions.
        
        Args:
            error: The original error
            operation: Description of the operation that failed
            
        Returns:
            Appropriate embedding exception
        """
        if isinstance(error, asyncio.TimeoutError):
            return APITimeoutError(f"Timeout during {operation}")
            
        elif hasattr(error, 'status') and error.status == 429:
            retry_after = getattr(error, 'retry_after', None)
            return RateLimitError(retry_after=retry_after, api_provider=self.config.provider)
            
        elif isinstance(error, (EmbeddingError, ValidationError)):
            # Already the right type
            return error
            
        else:
            # Generic API error
            return EmbeddingAPIError(
                f"API error during {operation}: {error}",
                api_provider=self.config.provider
            )

    async def _retry_with_backoff(self, operation_func, *args, **kwargs):
        """
        Execute an operation with exponential backoff retry logic.
        
        Args:
            operation_func: Async function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result of the operation
            
        Raises:
            EmbeddingError: If all retries are exhausted
        """
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return await operation_func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                
                # Convert to appropriate exception type
                embedding_error = self._handle_api_error(e, "embedding")
                
                # Don't retry on validation errors
                if isinstance(embedding_error, ValidationError):
                    raise embedding_error
                
                # If this was the last attempt, raise the error
                if attempt == self.config.max_retries:
                    raise embedding_error
                
                # Calculate backoff delay
                delay = self.config.retry_delay * (2 ** attempt)
                
                logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay}s: {embedding_error}"
                )
                
                await asyncio.sleep(delay)
        
        # This should never be reached, but just in case
        raise EmbeddingAPIError(f"All retry attempts exhausted. Last error: {last_error}") 