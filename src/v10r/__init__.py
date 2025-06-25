"""
v10r: Generic PostgreSQL NOTIFY-based vectorization service.

v10r automatically generates and maintains vector embeddings for any text content
in your PostgreSQL database using OpenAI-compatible embedding APIs.
"""

__version__ = "0.1.0"
__author__ = "v10r Contributors"
__email__ = "contributors@v10r.dev"

from .config import V10rConfig
from .exceptions import V10rError, ConfigurationError, DatabaseError, EmbeddingError

__all__ = [
    "__version__",
    "V10rConfig",
    "V10rError",
    "ConfigurationError", 
    "DatabaseError",
    "EmbeddingError",
] 