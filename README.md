# v10r

A PostgreSQL notify-based vectorization service that automatically creates and maintains vector embeddings for text data.

## Overview

v10r monitors PostgreSQL tables and automatically generates embeddings when text data changes. It handles schema management, real-time updates, and supports multiple embedding providers including Infinity Docker, OpenAI, and custom APIs.

## Features

- Automatic schema management and column creation
- Real-time processing via PostgreSQL NOTIFY/LISTEN
- Multiple embedding providers (Infinity, OpenAI, custom)
- Intelligent collision detection for column names
- Model version tracking and metadata
- Optimized batch processing
- Production-ready with Docker support

## Quick Start

### Installation

```bash
pip install v10r
```

### Initialize Configuration

```bash
v10r init --wizard
# Or use example config
v10r init --example > v10r-config.yaml
```

### Configure Your Table

```yaml
# v10r-config.yaml
databases:
  - name: "primary"
    connection:
      host: "${POSTGRES_HOST}"
      port: 5432
      database: "${POSTGRES_DB}" 
      user: "${POSTGRES_USER}"
      password: "${POSTGRES_PASSWORD}"

    tables:
      - schema: "public"
        table: "documents"
        text_column: "content"
        embedding_config: "infinity_bge_m3"

# Infinity configuration (BGE-M3)
embeddings:
  infinity_bge_m3:
    provider: "infinity"
    endpoint: "http://localhost:7997"
    model: "BAAI/bge-m3"
    dimensions: 1024
    batch_size: 30
```

### Start the Service

```bash
v10r register-table --config v10r-config.yaml --table public.documents
v10r start --config v10r-config.yaml
```

v10r will create the necessary columns, triggers, and begin processing your data.

## Infinity Docker

v10r works seamlessly with [Infinity](https://github.com/michaelfeil/infinity) for local embedding generation:

```bash
docker run -p 7997:7997 michaelf34/infinity:latest v2 --model-id BAAI/bge-m3
```

For production deployments, see the included `porter.yaml` configuration.

## Advanced Usage

### Multiple Columns Per Table

```yaml
tables:
  - schema: "public" 
    table: "products"
    text_column: "title"
    embedding_config: "infinity_bge_m3"
    
  - schema: "public"
    table: "products" 
    text_column: "description"
    embedding_config: "infinity_bge_m3"
    preprocessing:
      cleaning_config: "html_to_markdown"
```

### Text Preprocessing

```yaml
cleaning_configs:
  html_to_markdown:
    pipeline:
      - type: "html_extraction"
      - type: "markdown_conversion" 
      - type: "text_normalization"
```

### Schema Management

```yaml
schema_management:
  reconciliation: true
  collision_strategy: "auto_warn"
  allow_drop_columns: false
```

## Supported Providers

### Infinity Docker (Recommended)
- Models: BGE-M3, multilingual models
- Local or hosted deployment
- OpenAI-compatible API

### OpenAI
- Models: text-embedding-3-small, text-embedding-3-large
- Official API integration

### Custom APIs
- Any OpenAI-compatible endpoint
- Flexible configuration

## Architecture

```
PostgreSQL → NOTIFY Events → v10r Listener → Queue → Workers → Embedding API
```

The listener monitors database changes via PostgreSQL NOTIFY/LISTEN, queues tasks for processing, and workers generate embeddings in batches.

## Production Deployment

- Automatic schema reconciliation with collision detection
- Connection pooling and retry logic
- Health checks and Prometheus metrics
- Structured logging with correlation IDs
- Full audit trail and model version tracking

## Configuration Reference

### Database Connection
```yaml
databases:
  - name: "primary"
    connection:
      host: "${POSTGRES_HOST}"
      port: 5432
      database: "${POSTGRES_DB}"
      user: "${POSTGRES_USER}"
      password: "${POSTGRES_PASSWORD}"
      ssl_mode: "prefer" 
      connect_timeout: 30
```

### Embedding Providers
```yaml
embeddings:
  infinity_bge_m3:
    provider: "infinity"
    endpoint: "http://localhost:7997"
    model: "BAAI/bge-m3"
    dimensions: 1024
    batch_size: 30
    
  openai:
    provider: "openai"
    api_key: "${OPENAI_API_KEY}"
    model: "text-embedding-3-small"
    dimensions: 1536
```

### Table Configuration
```yaml
tables:
  - schema: "public"
    table: "documents"
    text_column: "content"
    embedding_config: "infinity_bge_m3"
    preprocessing:
      enabled: true
      cleaning_config: "html_to_markdown"
```

## CLI Commands

```bash
# Configuration
v10r init [--wizard] [--example]
v10r validate --config config.yaml

# Table management
v10r register-table --config config.yaml --table schema.table
v10r status --config config.yaml

# Schema operations
v10r schema reconcile --config config.yaml [--dry-run]

# Service control
v10r start --config config.yaml
v10r stop

# Health monitoring
v10r health --config config.yaml
```

## Development

For local development setup, see our [Development Guide](docs/development.md).

### Quick Start for Testing

```bash
# Set up development environment
make setup
source .venv/bin/activate  # REQUIRED: Always activate venv!
make install-dev

# Run tests
make test              # Unit tests only (fast)
make test-coverage     # With coverage report
make test-all          # All tests including integration
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Development setup
- Code style and standards  
- Commit message format
- Pull request process
- Testing requirements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- [GitHub Issues](https://github.com/randomm/v10r/issues)
- [GitHub Discussions](https://github.com/randomm/v10r/discussions)

## Documentation

- [Getting Started](docs/getting-started.md) - Quick setup guide
- [Configuration](docs/configuration.md) - YAML configuration reference
- [Architecture](docs/architecture.md) - System design
- [API Reference](docs/api-reference.md) - CLI commands and APIs
- [Development](docs/development.md) - Local development setup

## Roadmap

- [x] Core vectorization with PostgreSQL NOTIFY
- [x] Multiple embedding provider support
- [x] Automatic schema management
- [x] Text preprocessing pipelines
- [ ] Reranking model integration
- [ ] Multi-modal embeddings
- [ ] Kubernetes operator
- [ ] Web dashboard 