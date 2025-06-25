"""
Text cleaning implementations for preprocessing pipeline.

This module provides various text cleaning strategies from basic whitespace
normalization to HTML content extraction and markdown conversion.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BaseTextCleaner(ABC):
    """Abstract base class for text cleaning implementations."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        self.options = options or {}
    
    @abstractmethod
    async def clean(self, text: str) -> str:
        """Clean the input text and return processed result."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the cleaner name for tracking."""
        pass


class BasicTextCleaner(BaseTextCleaner):
    """Basic text cleaning for simple whitespace and unicode normalization."""
    
    @property 
    def name(self) -> str:
        return "basic_cleanup"
    
    async def clean(self, text: str) -> str:
        """Perform basic text cleaning operations."""
        if not text:
            return ""
        
        try:
            # Unicode normalization (fix common encoding issues)
            try:
                import ftfy
                text = ftfy.fix_text(text)
            except ImportError:
                logger.debug("ftfy not available, skipping unicode fixes")
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Remove null bytes and other control characters
            text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Basic text cleaning failed: {e}")
            return text  # Return original text if cleaning fails


class HTMLTextCleaner(BaseTextCleaner):
    """HTML content extraction and cleaning with markdown conversion."""
    
    @property
    def name(self) -> str:
        return "html_to_markdown"
    
    async def clean(self, text: str) -> str:
        """Extract content from HTML and convert to clean markdown."""
        if not text:
            return ""
        
        try:
            # First try trafilatura for high-quality content extraction
            cleaned_text = await self._extract_with_trafilatura(text)
            
            # If trafilatura fails, fall back to BeautifulSoup
            if not cleaned_text:
                cleaned_text = await self._extract_with_beautifulsoup(text)
            
            # Apply basic cleaning to the extracted text
            basic_cleaner = BasicTextCleaner()
            cleaned_text = await basic_cleaner.clean(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"HTML text cleaning failed: {e}")
            # Fall back to basic cleaning on error
            basic_cleaner = BasicTextCleaner()
            return await basic_cleaner.clean(text)
    
    async def _extract_with_trafilatura(self, html: str) -> str:
        """Extract main content using trafilatura."""
        try:
            import trafilatura
            
            # Extract main content
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                output_format='markdown'
            )
            
            return extracted or ""
            
        except ImportError:
            logger.debug("trafilatura not available, falling back to BeautifulSoup")
            return ""
        except Exception as e:
            logger.debug(f"trafilatura extraction failed: {e}")
            return ""
    
    async def _extract_with_beautifulsoup(self, html: str) -> str:
        """Extract text using BeautifulSoup as fallback."""
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up extracted text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            logger.warning("BeautifulSoup not available, returning original text")
            return html
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
            return html


class MarkdownTextCleaner(BaseTextCleaner):
    """Markdown content cleaning and normalization."""
    
    @property
    def name(self) -> str:
        return "markdown_cleanup"
    
    async def clean(self, text: str) -> str:
        """Clean and normalize markdown content."""
        if not text:
            return ""
        
        try:
            # Convert markdown to plain text while preserving structure
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Remove headers
            text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)  # Remove bold/italic
            text = re.sub(r'`([^`]+)`', r'\1', text)  # Remove inline code
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Extract link text
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Remove list markers
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered lists
            
            # Apply basic cleaning
            basic_cleaner = BasicTextCleaner()
            text = await basic_cleaner.clean(text)
            
            return text
            
        except Exception as e:
            logger.error(f"Markdown cleaning failed: {e}")
            return text 