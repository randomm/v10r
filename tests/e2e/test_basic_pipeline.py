"""
Basic end-to-end pipeline test for v10r vectorizer.

This test validates the complete pipeline from configuration to vector generation.
"""

import pytest
import tempfile
import yaml
from pathlib import Path

from v10r.config import V10rConfig
from v10r.embedding.factory import EmbeddingClientFactory


class TestBasicPipeline:
    """Basic end-to-end pipeline tests."""
    
    def test_config_loading_and_validation(self):
        """Test that configuration can be loaded and validated."""
        config_data = {
            'databases': [{
                'name': 'test_db',
                'connection': {
                    'host': 'localhost',
                    'port': 5432,
                    'database': 'test',
                    'user': 'test',
                    'password': 'test'
                },
                'tables': [{
                    'table': 'test_table',
                    'text_column': 'content',
                    'embedding_config': 'default'
                }]
            }],
            'embeddings': {
                'default': {
                    'provider': 'openai',
                    'api_key': 'sk-test-key',
                    'model': 'text-embedding-3-small',
                    'dimensions': 1536
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            # Test configuration loading
            config = V10rConfig.from_yaml(config_file)
            assert config is not None
            assert len(config.databases) == 1
            assert config.databases[0].name == 'test_db'
            assert 'default' in config.embeddings
            assert config.embeddings['default'].provider == 'openai'
            
        finally:
            Path(config_file).unlink()
    
    def test_embedding_client_factory_integration(self):
        """Test that embedding client factory works with configuration."""
        from v10r.config import EmbeddingConfig
        
        config = EmbeddingConfig(
            provider='openai',
            api_key='sk-test-key',
            model='text-embedding-3-small',
            dimensions=1536,
            batch_size=100,
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )
        
        # Test client creation
        client = EmbeddingClientFactory.create_client(config)
        assert client is not None
        assert client.model == 'text-embedding-3-small'
        assert client.config.dimensions == 1536
    
    def test_supported_providers_availability(self):
        """Test that all expected embedding providers are available."""
        providers = EmbeddingClientFactory.get_supported_providers()
        
        expected_providers = ['openai', 'infinity', 'custom']
        for provider in expected_providers:
            assert provider in providers, f"Provider {provider} not available"
    
    @pytest.mark.asyncio
    async def test_basic_embedding_pipeline(self):
        """Test basic embedding generation pipeline (mocked)."""
        from v10r.config import EmbeddingConfig
        from unittest.mock import AsyncMock, patch
        
        config = EmbeddingConfig(
            provider='openai',
            api_key='sk-test-key',
            model='text-embedding-3-small',
            dimensions=1536,
            batch_size=100,
            timeout=30,
            max_retries=3,
            retry_delay=1.0
        )
        
        client = EmbeddingClientFactory.create_client(config)
        
        # Mock the actual API call
        with patch.object(client, 'embed_texts', new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            
            # Test embedding generation
            texts = ["Hello world", "Test document"]
            embeddings = await client.embed_texts(texts)
            
            assert len(embeddings) == 2
            assert len(embeddings[0]) == 1536
            assert len(embeddings[1]) == 1536
            mock_embed.assert_called_once_with(texts)


if __name__ == '__main__':
    pytest.main([__file__]) 