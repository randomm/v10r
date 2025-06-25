"""
Custom API embedding client implementation.

This client can work with any OpenAI-compatible embedding API endpoint,
making it highly flexible for various self-hosted or third-party services.
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging
import aiohttp

from .base import EmbeddingClient
from ..config import EmbeddingConfig
from ..exceptions import EmbeddingError, EmbeddingAPIError, ValidationError, RateLimitError

logger = logging.getLogger(__name__)


class CustomAPIEmbeddingClient(EmbeddingClient):
    """
    Generic custom API embedding client for OpenAI-compatible endpoints.
    
    This client can work with any service that implements the OpenAI embeddings API:
    - Self-hosted embedding services
    - Third-party API providers
    - Custom embedding servers
    - Local development servers
    
    Configurable via environment variables for maximum flexibility.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize custom API client."""
        super().__init__(config)
        
        if not config.endpoint:
            raise ValidationError("Custom API endpoint is required")
            
        self.endpoint = config.endpoint.rstrip('/')
        self.api_key = config.api_key  # Optional for some services
        
        # Session for connection reuse
        self._session: aiohttp.ClientSession | None = None
        
        # Validate endpoint format
        self._validate_endpoint()
    
    def _validate_endpoint(self) -> None:
        """Validate custom endpoint format."""
        if not (self.endpoint.startswith('http://') or self.endpoint.startswith('https://')):
            raise ValidationError("Custom API endpoint must start with http:// or https://")
            
        # Warn about HTTP in production
        if self.endpoint.startswith('http://') and 'localhost' not in self.endpoint:
            logger.warning(
                "Using HTTP endpoint in production is not recommended. "
                "Consider using HTTPS for security."
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
            
            # Add authorization header if API key is provided
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
        return self._session
    
    def _get_embeddings_url(self) -> str:
        """
        Construct the embeddings URL.
        
        Assumes OpenAI-compatible API structure: /v1/embeddings
        """
        # Handle endpoints that already include the path
        if self.endpoint.endswith('/embeddings'):
            return self.endpoint
        elif '/v1/embeddings' in self.endpoint:
            return self.endpoint
        else:
            # Assume standard OpenAI structure
            return f"{self.endpoint}/v1/embeddings"
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts using custom API."""
        if not texts:
            return []
            
        self._validate_texts(texts)
        
        session = await self._get_session()
        
        # Use OpenAI-compatible request format
        payload = {
            "model": self.config.model,
            "input": texts,
        }
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                async with session.post(
                    self._get_embeddings_url(),
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Handle various response formats
                        embeddings = self._extract_embeddings(data)
                        
                        # Validate response
                        self._validate_embedding_response(embeddings, len(texts))
                        
                        logger.debug(
                            f"Successfully generated {len(embeddings)} embeddings "
                            f"using custom API {self.endpoint} with model {self.config.model}"
                        )
                        
                        return embeddings
                        
                    elif response.status == 429:
                        # Rate limit or overload
                        retry_after = int(response.headers.get("retry-after", 30))
                        if retry_count < self.config.max_retries:
                            logger.warning(
                                f"Custom API rate limit hit, retrying after {retry_after}s "
                                f"(attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(retry_after)
                            retry_count += 1
                            continue
                        else:
                            raise RateLimitError(retry_after=retry_after, api_provider="custom")
                            
                    elif response.status in [401, 403]:
                        # Authentication error - don't retry
                        try:
                            error_data = await response.json()
                            error_msg = self._extract_error_message(error_data)
                        except:
                            error_msg = "Authentication failed"
                        raise EmbeddingAPIError(
                            f"Custom API auth error: {error_msg}",
                            status_code=response.status,
                            api_provider="custom"
                        )
                        
                    elif response.status == 422:
                        # Validation error
                        try:
                            error_data = await response.json()
                            error_msg = self._extract_error_message(error_data)
                        except:
                            error_msg = "Validation error"
                        raise ValidationError(f"Custom API validation error: {error_msg}")
                        
                    else:
                        # Other API error
                        try:
                            error_data = await response.json()
                            error_msg = self._extract_error_message(error_data)
                        except:
                            error_msg = f"HTTP {response.status}"
                            
                        last_error = EmbeddingAPIError(
                            f"Custom API error: {error_msg}",
                            status_code=response.status,
                            api_provider="custom"
                        )
                        
                        if retry_count < self.config.max_retries:
                            delay = self.config.retry_delay * (2 ** retry_count)
                            logger.warning(
                                f"Custom API error (status {response.status}), "
                                f"retrying in {delay}s (attempt {retry_count + 1})"
                            )
                            await asyncio.sleep(delay)
                            retry_count += 1
                            continue
                        else:
                            raise last_error
                            
            except aiohttp.ClientError as e:
                last_error = EmbeddingAPIError(f"Network error connecting to custom API: {e}", api_provider="custom")
                
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
    
    def _extract_embeddings(self, response_data: Dict[str, Any]) -> List[List[float]]:
        """
        Extract embeddings from API response, handling various formats.
        
        Args:
            response_data: JSON response from the API
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingAPIError: If response format is unexpected
        """
        # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
        if "data" in response_data:
            return [item["embedding"] for item in response_data["data"]]
        
        # Direct embeddings format: {"embeddings": [[...], [...]]}
        elif "embeddings" in response_data:
            return response_data["embeddings"]
        
        # Simple array format: [[...], [...]]
        elif isinstance(response_data, list):
            return response_data
        
        # Hugging Face format: {"embeddings": [...]} or similar
        elif len(response_data) == 1:
            key = list(response_data.keys())[0]
            value = response_data[key]
            if isinstance(value, list) and value:
                # Check if it's a list of embeddings or a single embedding
                if isinstance(value[0], list):
                    return value
                else:
                    return [value]  # Single embedding
        
        else:
            raise EmbeddingAPIError(
                f"Unexpected response format. Available keys: {list(response_data.keys())}",
                api_provider="custom"
            )
    
    def _extract_error_message(self, error_data: Dict[str, Any]) -> str:
        """
        Extract error message from API error response, handling various formats.
        
        Args:
            error_data: JSON error response from the API
            
        Returns:
            Error message string
        """
        # OpenAI format: {"error": {"message": "...", "type": "...", ...}}
        if "error" in error_data:
            error = error_data["error"]
            if isinstance(error, dict) and "message" in error:
                return error["message"]
            elif isinstance(error, str):
                return error
        
        # Direct message format: {"message": "..."}
        elif "message" in error_data:
            return error_data["message"]
        
        # Detail format (FastAPI/Pydantic): {"detail": "..."}
        elif "detail" in error_data:
            detail = error_data["detail"]
            if isinstance(detail, str):
                return detail
            elif isinstance(detail, list) and detail:
                return str(detail[0])
        
        # Description format: {"description": "..."}
        elif "description" in error_data:
            return error_data["description"]
        
        # Fallback to string representation
        else:
            return str(error_data)
    
    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the custom API service."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            session = await self._get_session()
            
            # Try to get model info if available
            model_info = await self._try_get_models()
            
            # Test embedding generation
            test_embedding = await self.embed_single("health check")
            
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "provider": "custom",
                "model": self.config.model,
                "endpoint": self.endpoint,
                "dimensions": len(test_embedding),
                "api_accessible": True,
                "has_api_key": bool(self.api_key),
                "response_time_ms": round(response_time, 2),
                **model_info
            }
            
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return {
                "status": "unhealthy",
                "provider": "custom",
                "model": self.config.model,
                "endpoint": self.endpoint,
                "error": str(e),
                "api_accessible": False,
                "has_api_key": bool(self.api_key),
                "response_time_ms": round(response_time, 2)
            }
    
    async def _try_get_models(self) -> Dict[str, Any]:
        """
        Try to get available models from the API.
        
        Returns:
            Dictionary with model information, empty if not available
        """
        try:
            session = await self._get_session()
            
            # Try common model endpoints
            model_endpoints = [
                f"{self.endpoint}/v1/models",
                f"{self.endpoint}/models",
                f"{self.endpoint}/api/models"
            ]
            
            for url in model_endpoints:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            return {
                                "available_models": data.get("data", data),
                                "models_endpoint": url,
                                "models_accessible": True
                            }
                except:
                    continue
                    
            return {"models_accessible": False}
            
        except Exception:
            return {"models_accessible": False}
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None 