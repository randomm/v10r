# Contributing to v10r

Thank you for your interest in contributing to v10r! We welcome contributions from the community and are grateful for any help you can provide.

## How to Contribute

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
3. **Make your changes** with tests
4. **Run quality checks** locally
5. **Submit a pull request** with a clear description

## Development Setup

See our [Development Guide](docs/development.md) for detailed setup instructions.

## Code Style

We use modern Python practices and tooling:

- **Code formatting**: `ruff format` (automatically applied)
- **Linting**: `ruff check` with strict rules
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style for all public APIs
- **Test coverage**: Maintain >80% coverage

### Style Guidelines

- Follow PEP 8 with 88-character line limit (Black standard)
- Use meaningful variable and function names
- Keep functions small and focused (< 50 lines)
- Prefer composition over inheritance
- Use async/await for I/O operations

### Example Code Style

```python
from typing import List, Optional

async def process_embeddings(
    texts: List[str],
    model_name: str = "infinity_bge_m3",
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        model_name: Name of the embedding model to use
        batch_size: Optional batch size for processing
        
    Returns:
        List of embedding vectors
        
    Raises:
        EmbeddingError: If embedding generation fails
    """
    # Implementation here
    pass
```

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

### Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

### Examples
```bash
feat(embedding): add support for custom embedding providers

fix(worker): resolve memory leak in batch processing

docs(api): update configuration reference

test(schema): add edge case tests for collision detection

refactor(cli): simplify command argument parsing
```

## Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add tests** for new features or bug fixes
3. **Run the test suite** and ensure all tests pass
4. **Run quality checks** with `make check-all`
5. **Update the README** if adding new features
6. **Maintain backwards compatibility** when possible

### PR Title Format
Use the same convention as commit messages:
- `feat: add custom embedding provider support`
- `fix: resolve connection timeout issue`
- `docs: improve getting started guide`

### PR Description Template
```markdown
## Description
Brief description of the changes and why they're needed.

## Type of Change
- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature that breaks existing functionality)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for changes
- [ ] Existing tests updated as needed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

## Testing

### Writing Tests

- Write tests before implementing features (TDD)
- Use descriptive test names that explain what's being tested
- Follow the Arrange-Act-Assert pattern
- Mock external dependencies appropriately
- Include both positive and negative test cases

### Test Organization

```python
class TestEmbeddingClient:
    """Tests for embedding client functionality."""
    
    def test_successful_embedding_generation(self):
        """Test that embeddings are generated correctly."""
        # Arrange
        client = MockEmbeddingClient()
        texts = ["Hello world"]
        
        # Act
        embeddings = client.embed_batch(texts)
        
        # Assert
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 1536
    
    def test_handles_api_errors_gracefully(self):
        """Test that API errors are handled properly."""
        # Test implementation
```

## Reporting Issues

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version, v10r version)
- Relevant logs or error messages

### Feature Requests

Include:
- Clear use case
- Proposed solution
- Alternative solutions considered
- Impact on existing functionality

## Community

- Be respectful and inclusive
- Help others when you can
- Ask questions in discussions, not issues
- Follow our Code of Conduct

## Recognition

Contributors will be recognized in:
- Release notes for significant contributions
- Project documentation for sustained contributions
- GitHub contributors graph

Thank you for contributing to v10r! 🚀