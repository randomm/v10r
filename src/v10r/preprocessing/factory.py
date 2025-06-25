"""
Factory for creating text preprocessing pipelines based on configuration.

This module provides a factory pattern for creating and configuring
text cleaning pipelines from YAML configuration.
"""

import logging
from typing import Dict, Any, Optional, List

from ..config import CleaningConfig, CleaningPipelineStep
from .cleaners import BaseTextCleaner, BasicTextCleaner, HTMLTextCleaner, MarkdownTextCleaner

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """A pipeline of text cleaning steps."""
    
    def __init__(self, cleaners: List[BaseTextCleaner], name: str = "custom"):
        self.cleaners = cleaners
        self.name = name
    
    async def process(self, text: str) -> str:
        """Process text through all cleaning steps."""
        result = text
        
        for cleaner in self.cleaners:
            try:
                result = await cleaner.clean(result)
                logger.debug(f"Applied {cleaner.name} to text (length: {len(result)})")
            except Exception as e:
                logger.error(f"Cleaning step {cleaner.name} failed: {e}")
                # Continue with next cleaner, don't break the pipeline
        
        return result
    
    @property
    def config_name(self) -> str:
        """Get a string representation of the pipeline configuration."""
        cleaner_names = [cleaner.name for cleaner in self.cleaners]
        return f"{self.name}:{'->'.join(cleaner_names)}"


class PreprocessingFactory:
    """Factory for creating text preprocessing pipelines."""
    
    # Registry of available cleaners
    _CLEANER_REGISTRY = {
        "basic_cleanup": BasicTextCleaner,
        "html_extraction": HTMLTextCleaner,
        "html_to_markdown": HTMLTextCleaner,
        "markdown_cleanup": MarkdownTextCleaner,
        "text_normalization": BasicTextCleaner,
        "whitespace_cleanup": BasicTextCleaner,
    }
    
    @classmethod
    def create_pipeline(cls, config: CleaningConfig) -> PreprocessingPipeline:
        """Create a preprocessing pipeline from configuration."""
        cleaners = []
        
        for step in config.pipeline:
            cleaner = cls._create_cleaner_from_step(step)
            if cleaner:
                cleaners.append(cleaner)
        
        return PreprocessingPipeline(cleaners, config.name)
    
    @classmethod
    def create_simple_pipeline(cls, cleaner_name: str) -> PreprocessingPipeline:
        """Create a simple single-step pipeline."""
        cleaner_class = cls._CLEANER_REGISTRY.get(cleaner_name)
        
        if not cleaner_class:
            logger.warning(f"Unknown cleaner: {cleaner_name}, using basic cleanup")
            cleaner_class = BasicTextCleaner
        
        cleaner = cleaner_class()
        return PreprocessingPipeline([cleaner], cleaner_name)
    
    @classmethod
    def _create_cleaner_from_step(cls, step: CleaningPipelineStep) -> Optional[BaseTextCleaner]:
        """Create a cleaner instance from a pipeline step configuration."""
        try:
            # Map step types to cleaner classes
            step_type = step.type.lower()
            
            if step_type in ["html_extraction", "html"]:
                return HTMLTextCleaner(step.options)
            elif step_type in ["markdown_conversion", "markdown"]:
                return MarkdownTextCleaner(step.options)
            elif step_type in ["text_normalization", "normalization", "basic"]:
                return BasicTextCleaner(step.options)
            elif step_type in ["whitespace_cleanup", "whitespace"]:
                return BasicTextCleaner(step.options)
            else:
                # Try to find by exact name in registry
                cleaner_class = cls._CLEANER_REGISTRY.get(step_type)
                if cleaner_class:
                    return cleaner_class(step.options)
                else:
                    logger.warning(f"Unknown cleaning step type: {step_type}")
                    return None
        
        except Exception as e:
            logger.error(f"Failed to create cleaner for step {step.type}: {e}")
            return None
    
    @classmethod
    def get_default_html_pipeline(cls) -> PreprocessingPipeline:
        """Get a default pipeline for HTML content processing."""
        cleaners = [
            HTMLTextCleaner(),  # Extract content from HTML
            BasicTextCleaner()   # Basic text normalization
        ]
        return PreprocessingPipeline(cleaners, "html_to_text")
    
    @classmethod
    def get_default_markdown_pipeline(cls) -> PreprocessingPipeline:
        """Get a default pipeline for Markdown content processing."""
        cleaners = [
            MarkdownTextCleaner(),  # Clean markdown formatting
            BasicTextCleaner()       # Basic text normalization
        ]
        return PreprocessingPipeline(cleaners, "markdown_to_text")
    
    @classmethod
    def get_basic_pipeline(cls) -> PreprocessingPipeline:
        """Get a basic text cleaning pipeline."""
        cleaners = [BasicTextCleaner()]
        return PreprocessingPipeline(cleaners, "basic_text")


# Convenience function for creating preprocessors
async def preprocess_text(text: str, cleaning_config_name: str = "basic_cleanup") -> str:
    """Simple function to preprocess text with a named configuration."""
    pipeline = PreprocessingFactory.create_simple_pipeline(cleaning_config_name)
    return await pipeline.process(text) 