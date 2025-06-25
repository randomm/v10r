"""
Tests for text preprocessing functionality.

This module tests the text cleaning pipelines and factory functionality.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from v10r.preprocessing.cleaners import BaseTextCleaner, BasicTextCleaner, HTMLTextCleaner, MarkdownTextCleaner
from v10r.preprocessing.factory import PreprocessingFactory, PreprocessingPipeline, preprocess_text
from v10r.config import CleaningConfig, CleaningPipelineStep


class TestBasicTextCleaner:
    """Test BasicTextCleaner functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_name(self):
        """Test basic cleaner name property."""
        cleaner = BasicTextCleaner()
        assert cleaner.name == "basic_cleanup"
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_empty_text(self):
        """Test basic cleaner with empty text."""
        cleaner = BasicTextCleaner()
        result = await cleaner.clean("")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_whitespace_normalization(self):
        """Test whitespace normalization."""
        cleaner = BasicTextCleaner()
        text = "  Hello    world  \n\n  test  "
        result = await cleaner.clean(text)
        assert result == "Hello world test"
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_control_characters(self):
        """Test removal of control characters."""
        cleaner = BasicTextCleaner()
        text = "Hello\x00world\x08test\x7F"
        result = await cleaner.clean(text)
        assert result == "Helloworldtest"
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_with_ftfy(self):
        """Test basic cleaner with ftfy available."""
        with patch('builtins.__import__') as mock_import:
            # Mock ftfy module
            mock_ftfy = MagicMock()
            mock_ftfy.fix_text.return_value = "fixed text"
            
            def side_effect(name, *args, **kwargs):
                if name == 'ftfy':
                    return mock_ftfy
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = BasicTextCleaner()
            result = await cleaner.clean("broken text")
            assert result == "fixed text"
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_without_ftfy(self):
        """Test basic cleaner without ftfy (ImportError)."""
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'ftfy':
                    raise ImportError("No module named 'ftfy'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = BasicTextCleaner()
            result = await cleaner.clean("  test  text  ")
            assert result == "test text"
    
    @pytest.mark.asyncio
    async def test_basic_cleaner_exception_handling(self):
        """Test basic cleaner handles exceptions gracefully."""
        cleaner = BasicTextCleaner()
        
        # Mock re.sub to raise exception
        with patch('re.sub', side_effect=Exception("regex error")):
            result = await cleaner.clean("test text")
            # Should return original text on error
            assert result == "test text"


class TestHTMLTextCleaner:
    """Test HTMLTextCleaner functionality."""
    
    @pytest.mark.asyncio
    async def test_html_cleaner_name(self):
        """Test HTML cleaner name property."""
        cleaner = HTMLTextCleaner()
        assert cleaner.name == "html_to_markdown"
    
    @pytest.mark.asyncio
    async def test_html_cleaner_empty_text(self):
        """Test HTML cleaner with empty text."""
        cleaner = HTMLTextCleaner()
        result = await cleaner.clean("")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_html_cleaner_with_trafilatura(self):
        """Test HTML cleaner with trafilatura available."""
        with patch('builtins.__import__') as mock_import:
            # Mock trafilatura module
            mock_trafilatura = MagicMock()
            mock_trafilatura.extract.return_value = "extracted content"
            
            def side_effect(name, *args, **kwargs):
                if name == 'trafilatura':
                    return mock_trafilatura
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = HTMLTextCleaner()
            result = await cleaner.clean("<html><body>test</body></html>")
            
            assert result == "extracted content"
    
    @pytest.mark.asyncio
    async def test_html_cleaner_trafilatura_import_error(self):
        """Test HTML cleaner when trafilatura is not available."""
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'trafilatura':
                    raise ImportError("No module named 'trafilatura'")
                elif name == 'bs4':
                    # Mock BeautifulSoup
                    mock_bs4 = MagicMock()
                    mock_soup = MagicMock()
                    mock_soup.get_text.return_value = "extracted text"
                    mock_soup.return_value = []  # No scripts/styles to remove
                    mock_bs4.BeautifulSoup.return_value = mock_soup
                    return mock_bs4
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = HTMLTextCleaner()
            result = await cleaner.clean("<html><body>test</body></html>")
            
            assert "extracted text" in result
    
    @pytest.mark.asyncio
    async def test_html_cleaner_beautifulsoup_fallback(self):
        """Test HTML cleaner BeautifulSoup fallback."""
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'trafilatura':
                    # Mock trafilatura but return empty result to trigger fallback
                    mock_trafilatura = MagicMock()
                    mock_trafilatura.extract.return_value = None
                    return mock_trafilatura
                elif name == 'bs4':
                    # Mock BeautifulSoup
                    mock_bs4 = MagicMock()
                    mock_soup = MagicMock()
                    mock_soup.get_text.return_value = "Beautiful soup text"
                    mock_soup.return_value = []  # No scripts/styles to remove
                    mock_bs4.BeautifulSoup.return_value = mock_soup
                    return mock_bs4
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = HTMLTextCleaner()
            result = await cleaner.clean("<html><body>test</body></html>")
            
            assert "Beautiful soup text" in result
    
    @pytest.mark.asyncio
    async def test_html_cleaner_exception_handling(self):
        """Test HTML cleaner handles exceptions gracefully."""
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'trafilatura':
                    raise Exception("trafilatura error")
                elif name == 'bs4':
                    raise Exception("bs4 error")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            cleaner = HTMLTextCleaner()
            result = await cleaner.clean("<html>test</html>")
            # Should fall back to basic cleaning
            assert result == "<html>test</html>"


class TestMarkdownTextCleaner:
    """Test MarkdownTextCleaner functionality."""
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_name(self):
        """Test markdown cleaner name property."""
        cleaner = MarkdownTextCleaner()
        assert cleaner.name == "markdown_cleanup"
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_empty_text(self):
        """Test markdown cleaner with empty text."""
        cleaner = MarkdownTextCleaner()
        result = await cleaner.clean("")
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_headers(self):
        """Test markdown header removal."""
        cleaner = MarkdownTextCleaner()
        text = "# Header 1\n## Header 2\nContent"
        result = await cleaner.clean(text)
        assert "Header 1" in result
        assert "Header 2" in result
        assert "Content" in result
        assert "#" not in result
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_formatting(self):
        """Test markdown formatting removal."""
        cleaner = MarkdownTextCleaner()
        text = "**bold** and *italic* and `code`"
        result = await cleaner.clean(text)
        assert result == "bold and italic and code"
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_links(self):
        """Test markdown link text extraction."""
        cleaner = MarkdownTextCleaner()
        text = "Check out [this link](https://example.com) for more info"
        result = await cleaner.clean(text)
        assert "Check out this link for more info" in result
        assert "https://example.com" not in result
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_lists(self):
        """Test markdown list marker removal."""
        cleaner = MarkdownTextCleaner()
        text = "- Item 1\n* Item 2\n+ Item 3\n1. Numbered item"
        result = await cleaner.clean(text)
        assert "Item 1" in result
        assert "Item 2" in result
        assert "Item 3" in result
        assert "Numbered item" in result
        assert "-" not in result
        assert "*" not in result
        assert "1." not in result
    
    @pytest.mark.asyncio
    async def test_markdown_cleaner_exception_handling(self):
        """Test markdown cleaner handles exceptions gracefully."""
        cleaner = MarkdownTextCleaner()
        
        with patch('re.sub', side_effect=Exception("regex error")):
            result = await cleaner.clean("# Test")
            # Should return original text on error
            assert result == "# Test"


class TestPreprocessingPipeline:
    """Test PreprocessingPipeline functionality."""
    
    @pytest.mark.asyncio
    async def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        cleaners = [BasicTextCleaner(), HTMLTextCleaner()]
        pipeline = PreprocessingPipeline(cleaners, "test_pipeline")
        
        assert pipeline.cleaners == cleaners
        assert pipeline.name == "test_pipeline"
    
    @pytest.mark.asyncio
    async def test_pipeline_process(self):
        """Test pipeline text processing."""
        mock_cleaner1 = AsyncMock(spec=BaseTextCleaner)
        mock_cleaner1.clean.return_value = "step1_result"
        mock_cleaner1.name = "cleaner1"
        
        mock_cleaner2 = AsyncMock(spec=BaseTextCleaner)
        mock_cleaner2.clean.return_value = "step2_result"
        mock_cleaner2.name = "cleaner2"
        
        pipeline = PreprocessingPipeline([mock_cleaner1, mock_cleaner2])
        result = await pipeline.process("original_text")
        
        assert result == "step2_result"
        mock_cleaner1.clean.assert_called_once_with("original_text")
        mock_cleaner2.clean.assert_called_once_with("step1_result")
    
    @pytest.mark.asyncio
    async def test_pipeline_process_with_exception(self):
        """Test pipeline continues processing even if one cleaner fails."""
        mock_cleaner1 = AsyncMock(spec=BaseTextCleaner)
        mock_cleaner1.clean.side_effect = Exception("cleaner1 failed")
        mock_cleaner1.name = "cleaner1"
        
        mock_cleaner2 = AsyncMock(spec=BaseTextCleaner)
        mock_cleaner2.clean.return_value = "step2_result"
        mock_cleaner2.name = "cleaner2"
        
        pipeline = PreprocessingPipeline([mock_cleaner1, mock_cleaner2])
        result = await pipeline.process("original_text")
        
        # Should continue with original text to cleaner2
        assert result == "step2_result"
        mock_cleaner2.clean.assert_called_once_with("original_text")
    
    def test_pipeline_config_name(self):
        """Test pipeline config name generation."""
        mock_cleaner1 = MagicMock(spec=BaseTextCleaner)
        mock_cleaner1.name = "basic_cleanup"
        
        mock_cleaner2 = MagicMock(spec=BaseTextCleaner)
        mock_cleaner2.name = "html_to_markdown"
        
        pipeline = PreprocessingPipeline([mock_cleaner1, mock_cleaner2], "test")
        assert pipeline.config_name == "test:basic_cleanup->html_to_markdown"


class TestPreprocessingFactory:
    """Test PreprocessingFactory functionality."""
    
    def test_create_simple_pipeline(self):
        """Test creating a simple pipeline."""
        pipeline = PreprocessingFactory.create_simple_pipeline("basic_cleanup")
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 1
        assert isinstance(pipeline.cleaners[0], BasicTextCleaner)
        assert pipeline.name == "basic_cleanup"
    
    def test_create_simple_pipeline_unknown_cleaner(self):
        """Test creating pipeline with unknown cleaner name."""
        pipeline = PreprocessingFactory.create_simple_pipeline("unknown_cleaner")
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 1
        assert isinstance(pipeline.cleaners[0], BasicTextCleaner)  # Falls back to basic
    
    def test_create_pipeline_from_config(self):
        """Test creating pipeline from configuration."""
        steps = [
            CleaningPipelineStep(type="html_extraction", options={}),
            CleaningPipelineStep(type="basic", options={})
        ]
        config = CleaningConfig(name="test_config", pipeline=steps)
        
        pipeline = PreprocessingFactory.create_pipeline(config)
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 2
        assert isinstance(pipeline.cleaners[0], HTMLTextCleaner)
        assert isinstance(pipeline.cleaners[1], BasicTextCleaner)
        assert pipeline.name == "test_config"
    
    def test_get_default_html_pipeline(self):
        """Test getting default HTML pipeline."""
        pipeline = PreprocessingFactory.get_default_html_pipeline()
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 2
        assert isinstance(pipeline.cleaners[0], HTMLTextCleaner)
        assert isinstance(pipeline.cleaners[1], BasicTextCleaner)
        assert pipeline.name == "html_to_text"
    
    def test_get_default_markdown_pipeline(self):
        """Test getting default Markdown pipeline."""
        pipeline = PreprocessingFactory.get_default_markdown_pipeline()
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 2
        assert isinstance(pipeline.cleaners[0], MarkdownTextCleaner)
        assert isinstance(pipeline.cleaners[1], BasicTextCleaner)
        assert pipeline.name == "markdown_to_text"
    
    def test_get_basic_pipeline(self):
        """Test getting basic pipeline."""
        pipeline = PreprocessingFactory.get_basic_pipeline()
        
        assert isinstance(pipeline, PreprocessingPipeline)
        assert len(pipeline.cleaners) == 1
        assert isinstance(pipeline.cleaners[0], BasicTextCleaner)
        assert pipeline.name == "basic_text"
    
    def test_create_cleaner_from_step_html(self):
        """Test creating HTML cleaner from step."""
        step = CleaningPipelineStep(type="html_extraction", options={})
        cleaner = PreprocessingFactory._create_cleaner_from_step(step)
        
        assert isinstance(cleaner, HTMLTextCleaner)
    
    def test_create_cleaner_from_step_markdown(self):
        """Test creating Markdown cleaner from step."""
        step = CleaningPipelineStep(type="markdown_conversion", options={})
        cleaner = PreprocessingFactory._create_cleaner_from_step(step)
        
        assert isinstance(cleaner, MarkdownTextCleaner)
    
    def test_create_cleaner_from_step_basic(self):
        """Test creating basic cleaner from step."""
        step = CleaningPipelineStep(type="text_normalization", options={})
        cleaner = PreprocessingFactory._create_cleaner_from_step(step)
        
        assert isinstance(cleaner, BasicTextCleaner)
    
    def test_create_cleaner_from_step_unknown(self):
        """Test creating cleaner from unknown step type."""
        step = CleaningPipelineStep(type="unknown_type", options={})
        cleaner = PreprocessingFactory._create_cleaner_from_step(step)
        
        assert cleaner is None
    
    def test_create_cleaner_from_step_exception(self):
        """Test creating cleaner handles exceptions."""
        step = CleaningPipelineStep(type="basic", options={})
        
        with patch('v10r.preprocessing.factory.BasicTextCleaner', side_effect=Exception("creation failed")):
            cleaner = PreprocessingFactory._create_cleaner_from_step(step)
            assert cleaner is None


class TestConvenienceFunction:
    """Test the convenience preprocess_text function."""
    
    @pytest.mark.asyncio
    async def test_preprocess_text_default(self):
        """Test preprocess_text with default configuration."""
        result = await preprocess_text("  test  text  ")
        assert result == "test text"
    
    @pytest.mark.asyncio
    async def test_preprocess_text_custom_config(self):
        """Test preprocess_text with custom configuration."""
        result = await preprocess_text("**bold text**", "markdown_cleanup")
        assert result == "bold text"
        assert "**" not in result 