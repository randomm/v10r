# Development Guide

This guide helps you set up v10r for local development and testing.

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- PostgreSQL 14+ with pgvector extension
- Redis (for message queuing)
- Docker (for local development stack)

## Quick Setup

### 1. Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone https://github.com/randomm/v10r.git
cd v10r

# Create virtual environment with uv
make setup

# IMPORTANT: Activate the virtual environment!
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make install-dev
```

### 3. Start Development Stack

```bash
# Start PostgreSQL and Redis using Docker
make dev-stack-up

# Verify services are running
docker ps
```

### 4. Run Tests

**IMPORTANT**: Always ensure your virtual environment is activated before running tests!

```bash
# Verify virtual environment is active
which python  # Should show .venv/bin/python

# Run unit tests (fast)
make test

# Run tests with coverage report
make test-coverage

# Run all tests including integration
make test-all

# Run tests in watch mode
make test-watch
```

#### Running Specific Tests

```bash
# Run a specific test file
python -m pytest tests/unit/test_config.py -v

# Run tests matching a pattern
python -m pytest -k "test_embedding" -v

# Run tests with specific markers
python -m pytest -m "not integration" -v
```

#### Troubleshooting Test Issues

**Tests timing out or hanging?**
- Ensure Docker services are running: `docker ps`
- Check virtual environment is activated: `echo $VIRTUAL_ENV`
- Run with verbose output: `pytest -xvs`

**Module import errors?**
- You're likely running pytest from system Python instead of venv
- Always use `python -m pytest` or activate venv first

**Coverage below 80%?**
- Run `make test-coverage` to see which lines are missing
- Focus on testing error paths and edge cases

### 5. Code Quality

```bash
# Run all quality checks
make check-all

# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

## Development Workflow

### Project Structure

```
v10r/
├── src/v10r/                 # Main package
│   ├── __init__.py
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration system
│   ├── exceptions.py        # Exception hierarchy
│   ├── listener.py          # PostgreSQL NOTIFY listener
│   ├── worker.py            # Background worker
│   ├── registration.py      # Table registration
│   ├── database/            # Database layer
│   ├── embedding/           # Embedding providers
│   ├── schema/              # Schema management
│   └── preprocessing/       # Text preprocessing
├── tests/                   # Test suite
├── examples/                # Example configurations
├── docs/                    # Documentation
├── pyproject.toml          # Project configuration
├── Makefile                # Development commands
└── README.md               # Project overview
```

### Configuration Testing

```bash
# Initialize a new config file
uv run v10r init --output test-config.yaml

# Validate configuration
uv run v10r validate-config --config test-config.yaml

# Test with example config
uv run v10r validate-config --config examples/v10r-config.yaml
```

### Adding New Features

When adding new features:

1. **Write tests first** (TDD approach)
2. **Update configuration models** if needed
3. **Add CLI commands** for new functionality
4. **Update documentation**
5. **Run quality checks** before committing

Example workflow:

```bash
# Create feature branch
git checkout -b feature/new-provider

# Write tests
touch tests/unit/test_new_provider.py

# Implement feature
touch src/v10r/embedding/new_provider.py

# Run tests during development
make test-watch

# Check code quality
make check-all

# Commit changes
git add .
git commit -m "feat: add new embedding provider"
```

## Environment Variables

Create a `.env` file for local development:

```bash
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=v10r
POSTGRES_USER=postgres
POSTGRES_PASSWORD=v10r

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Embedding APIs
OPENAI_API_KEY=your_openai_api_key
INFINITY_ENDPOINT=http://localhost:7997
CUSTOM_EMBEDDING_ENDPOINT=your_custom_endpoint
CUSTOM_API_KEY=your_custom_key
```

## Testing Strategy

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)

### Integration Tests
- Test component interactions
- Use test database and Redis
- Medium execution time (< 10 seconds per test)

### End-to-End Tests
- Test complete workflows
- Use real services in containers
- Slower execution (< 1 minute per test)

### Running Specific Tests

```bash
# Run specific test file
pytest tests/unit/test_config.py -v

# Run tests matching pattern
pytest -k "test_config" -v

# Run tests with specific markers
pytest -m "unit" -v

# Run tests with coverage for specific module
pytest --cov=src/v10r/config tests/unit/test_config.py
```

## Debugging

### Enable Debug Mode

```bash
# CLI debug mode
uv run v10r --debug validate-config --config test-config.yaml

# Python debug mode
export V10R_DEBUG=true
uv run v10r validate-config --config test-config.yaml
```

### Database Debugging

```bash
# Connect to development database
psql postgresql://postgres:v10r@localhost:5432/v10r

# Check pgvector extension
\dx

# List tables
\dt
```

### Redis Debugging

```bash
# Connect to Redis
redis-cli

# Monitor commands
redis-cli monitor

# Check queue contents
redis-cli llen v10r_tasks
```

## Performance Considerations

- Use `asyncio` for I/O operations
- Implement connection pooling
- Batch database operations
- Cache embedding results when possible
- Monitor memory usage with large datasets

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure virtual environment is activated
2. **Database connection**: Check PostgreSQL is running and accessible
3. **Redis connection**: Verify Redis service is running
4. **Permission errors**: Ensure proper database permissions
5. **API errors**: Verify API keys and endpoints are correct

### Getting Help

- Check the [documentation](index.md)
- Look at [example configurations](../examples/)
- Search [existing issues](https://github.com/randomm/v10r/issues)
- Ask questions in [discussions](https://github.com/randomm/v10r/discussions)