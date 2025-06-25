"""
OpenAI embedding client implementation.
"""

import asyncio
from typing import List, Dict, Any
import logging
import aiohttp

from .base import EmbeddingClient
from ..config import EmbeddingConfig
from ..exceptions import EmbeddingError, EmbeddingAPIError, ValidationError, RateLimitError

logger = logging.getLogger(__name__)


class OpenAIEmbeddingClient(EmbeddingClient):
    """
    OpenAI embedding client for text-embedding models.
    
    Supports OpenAI's text-embedding-3-small, text-embedding-3-large,
    and text-embedding-ada-002 models.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI client."""
        super().__init__(config)
        
        if not config.api_key:
            raise ValidationError("OpenAI API key is required")
            
        self.api_key = config.api_key
        self.base_url = "https://api.openai.com/v1"
        
        # Session for connection reuse
        self._session: aiohttp.ClientSession | None = None
        
        # Validate model is supported
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate that the model is supported by OpenAI."""
        supported_models = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        
        if self.config.model not in supported_models:
            logger.warning(
                f"Model {self.config.model} is not in known supported models: "
                f"{list(supported_models.keys())}. Proceeding anyway..."
            )
        elif supported_models[self.config.model] != self.config.dimensions:
            # For text-embedding-3 models, custom dimensions are supported
            if self.config.model.startswith("text-embedding-3"):
                if self.config.dimensions > supported_models[self.config.model]:
                    raise ValidationError(
                        f"Dimensions {self.config.dimensions} exceed maximum "
                        f"{supported_models[self.config.model]} for {self.config.model}"
                    )
            else:
                logger.warning(
                    f"Expected {supported_models[self.config.model]} dimensions "
                    f"for {self.config.model}, got {self.config.dimensions}"
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
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "User-Agent": "v10r-vectorizer/1.0"
                }
            )
        return self._session
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using OpenAI API."""
        if not texts:
            return []
            
        self._validate_texts(texts)
        
        session = await self._get_session()
        
        # Prepare request payload
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        
        # Add dimensions parameter for text-embedding-3 models if not default
        if (self.config.model.startswith("text-embedding-3") and 
            self.config.dimensions not in [1536, 3072]):
            payload["dimensions"] = self.config.dimensions
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                async with session.post(
                    f"{self.base_url}/embeddings",
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        embeddings = [item["embedding"] for item in data["data"]]
                        
                        # Validate response
                        self._validate_embedding_response(embeddings, len(texts))
                        
                        logger.debug(
                            f"Successfully generated {len(embeddings)} embeddings "
                            f"using model {self.config.model}"
                        )
                        
                        return embeddings
                        
                    elif response.status == 429:
                        # Rate limit hit
                        retry_after = int(response.headers.get("retry-after", 60))
                        if retry_count < self.config.max_retries:
                            logger.warning(
                                f"Rate limit hit, retrying after {retry_after}s "
                                f"(attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue
                        else:
                            raise RateLimitError(retry_after=retry_after, api_provider="openai")
                            
                    elif response.status in [401, 403]:
                        # Authentication error - don't retry
                        error_data = await response.json()
                        raise EmbeddingAPIError(
                            f"Authentication failed: {error_data.get('error', {}).get('message', 'Unknown error')}",
                            status_code=response.status,
                            api_provider="openai"
                        )
                        
                    else:
                        # Other API error
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("error", {}).get("message", "Unknown error")
                        except:
                            error_msg = f"HTTP {response.status}"
                            
                        last_error = EmbeddingAPIError(
                            f"OpenAI API error: {error_msg}",
                            status_code=response.status,
                            api_provider="openai"
                        )
                        
                        if retry_count < self.config.max_retries:
                            delay = self.config.retry_delay * (2 ** retry_count)
                            logger.warning(
                                f"API error (status {response.status}), retrying in {delay}s "
                                f"(attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(delay)
                            retry_count += 1
                            continue
                        else:
                            raise last_error
                            
            except aiohttp.ClientError as e:
                last_error = EmbeddingAPIError(f"Network error: {e}", api_provider="openai")
                
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the OpenAI API."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use a simple test embedding to check API health
            test_embedding = await self.embed_single("health check")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "provider": "openai",
                "model": self.config.model,
                "dimensions": len(test_embedding),
                "api_accessible": True,
                "response_time_ms": round(response_time, 2)
            }
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "unhealthy",
                "provider": "openai",
                "model": self.config.model,
                "error": str(e),
                "api_accessible": False,
                "response_time_ms": round(response_time, 2)
            }
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None 