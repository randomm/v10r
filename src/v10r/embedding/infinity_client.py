"""
Infinity Docker embedding client implementation.
"""

import asyncio
from typing import List, Dict, Any
import logging
import aiohttp

from .base import EmbeddingClient
from ..config import EmbeddingConfig
from ..exceptions import EmbeddingError, EmbeddingAPIError, ValidationError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


class InfinityEmbeddingClient(EmbeddingClient):
    """
    Infinity Docker embedding client for self-hosted models.
    
    Infinity is a high-performance embedding server that provides OpenAI-compatible
    API endpoints. It's particularly well-suited for production deployments with
    support for static embeddings and efficient resource utilization.
    
    Supported models include:
    - minishlab/potion-multilingual-128M (101 languages, 768d)
    - BAAI/bge-m3 (multilingual, 1024d)
    - sentence-transformers models
    - And many others supported by Infinity
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Infinity client."""
        super().__init__(config)
        
        if not config.endpoint:
            raise ValidationError("Infinity endpoint is required")
            
        self.endpoint = config.endpoint.rstrip('/')
        
        # Session for connection reuse
        self._session: aiohttp.ClientSession | None = None
        
        # Common model configurations for validation
        self._known_models = {
            "minishlab/potion-multilingual-128M": 768,
            "BAAI/bge-m3": 1024,
            "BAAI/bge-base-en-v1.5": 768,
            "BAAI/bge-large-en-v1.5": 1024,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }
        
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate model configuration against known models."""
        if self.config.model in self._known_models:
            expected_dim = self._known_models[self.config.model]
            if self.config.dimensions != expected_dim:
                logger.warning(
                    f"Expected {expected_dim} dimensions for {self.config.model}, "
                    f"got {self.config.dimensions}. This may be intentional for custom fine-tuning."
                )
        else:
            logger.info(
                f"Using custom or unknown model {self.config.model}. "
                f"Assuming {self.config.dimensions} dimensions."
            )
    
    def _validate_texts(self, texts: List[str]) -> None:
        """Validate input texts."""
        if not texts:
            raise ValidationError("Cannot embed empty list of texts")
            
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationError(f"Text at index {i} is not a string: {type(text)}")
    
    def _validate_embedding_response(self, embeddings: List[List[float]], expected_count: int) -> None:
        """Validate the embedding response."""
        if len(embeddings) != expected_count:
            raise EmbeddingError(
                f"Expected {expected_count} embeddings, got {len(embeddings)}"
            )
            
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.config.dimensions:
                raise EmbeddingError(
                    f"Embedding {i} has {len(embedding)} dimensions, "
                    f"expected {self.config.dimensions}"
                )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "v10r-vectorizer/1.0"
            }
            
            # Add API key if provided (some Infinity deployments require it)
            if self.config.api_key:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using Infinity API."""
        if not texts:
            return []
            
        self._validate_texts(texts)
        
        session = await self._get_session()
        
        # Infinity uses OpenAI-compatible API
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # Infinity typically uses /v1/embeddings endpoint
                url = f"{self.endpoint}/v1/embeddings"
                
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle both OpenAI format and potential Infinity variations
                        if "data" in data:
                            # OpenAI-compatible format
                            embeddings = [item["embedding"] for item in data["data"]]
                        elif "embeddings" in data:
                            # Direct embeddings format
                            embeddings = data["embeddings"]
                        else:
                            raise EmbeddingAPIError(f"Unexpected response format: {list(data.keys())}", api_provider="infinity")
                        
                        # Validate response
                        self._validate_embedding_response(embeddings, len(texts))
                        
                        logger.debug(
                            f"Successfully generated {len(embeddings)} embeddings "
                            f"using Infinity model {self.config.model}"
                        )
                        
                        return embeddings
                        
                    elif response.status == 429:
                        # Rate limit or overload
                        retry_after = int(response.headers.get("retry-after", 30))
                        if retry_count < self.config.max_retries:
                            logger.warning(
                                f"Infinity server overloaded, retrying after {retry_after}s "
                                f"(attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue
                        else:
                            raise RateLimitError(retry_after=retry_after, api_provider="infinity")
                            
                    elif response.status in [401, 403]:
                        # Authentication error - don't retry
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("detail", "Authentication failed")
                        except:
                            error_msg = "Authentication failed"
                        raise EmbeddingAPIError(
                            f"Infinity auth error: {error_msg}",
                            status_code=response.status,
                            api_provider="infinity"
                        )
                        
                    elif response.status == 422:
                        # Validation error (bad model, input too long, etc.)
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("detail", "Validation error")
                        except:
                            error_msg = "Validation error"
                        raise ValidationError(f"Infinity validation error: {error_msg}")
                        
                    else:
                        # Other API error
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("detail", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}"
                            
                        last_error = EmbeddingAPIError(
                            f"Infinity API error: {error_msg}",
                            status_code=response.status,
                            api_provider="infinity"
                        )
                        
                        if retry_count < self.config.max_retries:
                            delay = self.config.retry_delay * (2 ** retry_count)
                            logger.warning(
                                f"Infinity API error (status {response.status}), "
                                f"retrying in {delay}s (attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(delay)
                            retry_count += 1
                            continue
                        else:
                            raise last_error
                            
            except aiohttp.ClientError as e:
                # Check if it's specifically a timeout error
                if isinstance(e, (aiohttp.ServerTimeoutError, aiohttp.ClientTimeout, asyncio.TimeoutError)):
                    last_error = APITimeoutError(
                        message=f"Timeout connecting to Infinity: {e}",
                        timeout_duration=self.config.timeout,
                        api_provider="infinity"
                    )
                else:
                    last_error = EmbeddingAPIError(f"Network error connecting to Infinity: {e}", api_provider="infinity")
                
                if retry_count < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** retry_count)
                    logger.warning(
                        f"Network error, retrying in {delay}s (attempt {retry_count + 1}): {e}"
                    )
                    await asyncio.sleep(delay)
                    retry_count += 1
                    continue
                else:
                    raise last_error
        
        # Should not reach here, but just in case
        raise last_error or EmbeddingError("Failed to generate embeddings after all retries")
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Override batch processing for Infinity optimizations.
        
        Infinity performs better with consistent batch sizes, so we use
        the configured batch_size consistently rather than processing
        small final batches.
        """
        if not texts:
            return []
            
        self._validate_texts(texts)
        
        # For Infinity, always use the configured batch size for consistency
        # This helps with model optimization and resource utilization
        batch_size = self.config.batch_size
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = await self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Small delay between batches to prevent overwhelming the service
            if i + batch_size < len(texts):
                await asyncio.sleep(0.05)  # Shorter delay for local Infinity
                
        return all_embeddings
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the Infinity service."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            session = await self._get_session()
            
            # Try to get model info first
            model_info = {}
            try:
                async with session.get(f"{self.endpoint}/v1/models") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        model_info = {
                            "available_models": models_data.get("data", []),
                            "model_endpoint_accessible": True
                        }
            except Exception:
                model_info = {"model_endpoint_accessible": False}
            
            # Test embedding generation
            test_embedding = await self.embed_single("health check")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "provider": "infinity",
                "model": self.config.model,
                "endpoint": self.endpoint,
                "dimensions": len(test_embedding),
                "api_accessible": True,
                "response_time_ms": round(response_time, 2),
                **model_info
            }
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "unhealthy",
                "provider": "infinity",
                "model": self.config.model,
                "endpoint": self.endpoint,
                "error": str(e),
                "api_accessible": False,
                "response_time_ms": round(response_time, 2)
            }
    
    async def get_available_models(self) -> List[str]:
        """
        Get list of available models from Infinity.
        
        Returns:
            List of model names
        """
        try:
            session = await self._get_session()
            async with session.get(f"{self.endpoint}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    return [model.get("id", "") for model in models if model.get("id")]
                else:
                    raise EmbeddingAPIError(
                        f"Failed to get models: HTTP {response.status}",
                        status_code=response.status,
                        api_provider="infinity"
                    )
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            raise
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None 