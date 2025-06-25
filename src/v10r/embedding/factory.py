"""
Embedding client factory for creating appropriate clients based on configuration.
"""

from typing import Dict, Type
import logging

from .base import EmbeddingClient
from .openai_client import OpenAIEmbeddingClient
from .infinity_client import InfinityEmbeddingClient
from .custom_client import CustomAPIEmbeddingClient
from ..config import EmbeddingConfig
from ..exceptions import ValidationError, ConfigurationError


logger = logging.getLogger(__name__)


class EmbeddingClientFactory:
    """
    Factory for creating embedding clients based on provider configuration.
    
    This factory handles the instantiation of different embedding clients
    with primary focus on Infinity Docker deployments.
    """
    
    # Registry of available client implementations
    _CLIENT_REGISTRY: Dict[str, Type[EmbeddingClient]] = {
        "openai": OpenAIEmbeddingClient,
        "infinity": InfinityEmbeddingClient,
        "custom": CustomAPIEmbeddingClient,
    }
    
    @classmethod
    def create_client(cls, config: EmbeddingConfig) -> EmbeddingClient:
        """
        Create an embedding client based on the provider configuration.
        
        Args:
            config: Embedding configuration specifying the provider and settings
            
        Returns:
            Initialized embedding client
            
        Raises:
            ValidationError: If the provider is not supported
            ConfigurationError: If the configuration is invalid for the provider
        """
        provider = config.provider.lower()
        
        if provider not in cls._CLIENT_REGISTRY:
            available_providers = list(cls._CLIENT_REGISTRY.keys())
            raise ValidationError(
                f"Unsupported embedding provider: {provider}. "
                f"Available providers: {available_providers}"
            )
        
        client_class = cls._CLIENT_REGISTRY[provider]
        
        try:
            logger.info(f"Creating {provider} embedding client with model {config.model}")
            client = client_class(config)
            logger.debug(f"Successfully created {client}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to create {provider} client: {e}")
            raise ConfigurationError(
                f"Failed to create {provider} embedding client: {e}"
            ) from e
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """
        Get a list of supported embedding providers.
        
        Returns:
            List of provider names
        """
        return list(cls._CLIENT_REGISTRY.keys())
    
    @classmethod
    def register_provider(
        cls, 
        provider_name: str, 
        client_class: Type[EmbeddingClient]
    ) -> None:
        """
        Register a custom embedding client provider.
        
        This allows for extending the factory with custom embedding clients
        without modifying the core code.
        
        Args:
            provider_name: Name of the provider
            client_class: Client class that implements EmbeddingClient
            
        Raises:
            ValidationError: If the client class doesn't implement EmbeddingClient
        """
        if not issubclass(client_class, EmbeddingClient):
            raise ValidationError(
                f"Client class {client_class} must inherit from EmbeddingClient"
            )
        
        cls._CLIENT_REGISTRY[provider_name.lower()] = client_class
        logger.info(f"Registered custom embedding provider: {provider_name}")
    
    @classmethod
    def validate_config_for_provider(cls, config: EmbeddingConfig) -> bool:
        """
        Validate configuration for a specific provider without creating the client.
        
        Args:
            config: Embedding configuration to validate
            
        Returns:
            True if configuration is valid
            
        Raises:
            ValidationError: If configuration is invalid
        """
        provider = config.provider.lower()
        
        if provider not in cls._CLIENT_REGISTRY:
            available_providers = list(cls._CLIENT_REGISTRY.keys())
            raise ValidationError(
                f"Unsupported embedding provider: {provider}. "
                f"Available providers: {available_providers}"
            )
        
        # Provider-specific validation
        if provider == "openai":
            if not config.api_key:
                raise ValidationError("OpenAI provider requires api_key")
                
        elif provider == "infinity":
            if not config.endpoint:
                raise ValidationError("Infinity provider requires endpoint")
                
        elif provider == "custom":
            if not config.endpoint:
                raise ValidationError("Custom provider requires endpoint")
        
        # Common validation
        if config.dimensions <= 0:
            raise ValidationError("Dimensions must be positive")
            
        if config.batch_size <= 0:
            raise ValidationError("Batch size must be positive")
            
        if config.timeout <= 0:
            raise ValidationError("Timeout must be positive")
        
        return True
    
    @classmethod
    def get_provider_requirements(cls, provider: str) -> Dict[str, bool]:
        """
        Get the configuration requirements for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary indicating which configuration fields are required
            
        Raises:
            ValidationError: If provider is not supported
        """
        provider = provider.lower()
        
        if provider not in cls._CLIENT_REGISTRY:
            available_providers = list(cls._CLIENT_REGISTRY.keys())
            raise ValidationError(
                f"Unsupported embedding provider: {provider}. "
                f"Available providers: {available_providers}"
            )
        
        requirements = {
            "openai": {
                "api_key": True,
                "endpoint": False,
                "model": True,
                "dimensions": True,
            },
            "infinity": {
                "api_key": False,
                "endpoint": True,
                "model": True,
                "dimensions": True,
            },
            "custom": {
                "api_key": False,
                "endpoint": True,
                "model": True,
                "dimensions": True,
            },
        }
        
        return requirements[provider]
    
    @classmethod
    async def test_client_connectivity(cls, config: EmbeddingConfig) -> Dict[str, any]:
        """
        Test connectivity for an embedding client without storing it.
        
        Args:
            config: Embedding configuration to test
            
        Returns:
            Dictionary with connectivity test results
        """
        try:
            # Validate configuration first
            cls.validate_config_for_provider(config)
            
            # Create client and test connectivity
            async with cls.create_client(config) as client:
                health_result = await client.health_check()
                
            return {
                "connectivity_test": "passed",
                "provider": config.provider,
                "model": config.model,
                **health_result
            }
            
        except Exception as e:
            return {
                "connectivity_test": "failed",
                "provider": config.provider,
                "model": config.model,
                "error": str(e),
                "error_type": type(e).__name__
            } 