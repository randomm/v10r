"""
Embedding client system for v10r.

This package provides a unified interface for different embedding providers,
with primary focus on Infinity Docker deployments and OpenAI API compatibility.
"""

from .base import EmbeddingClient
from .openai_client import OpenAIEmbeddingClient
from .infinity_client import InfinityEmbeddingClient
from .custom_client import CustomAPIEmbeddingClient
from .factory import EmbeddingClientFactory

__all__ = [
    "EmbeddingClient",
    "OpenAIEmbeddingClient", 
    "InfinityEmbeddingClient",
    "CustomAPIEmbeddingClient", 
    "EmbeddingClientFactory",
] 