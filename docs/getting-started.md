# Getting Started with v10r

Get your first text vectorization running in under 5 minutes.

## Prerequisites

- PostgreSQL 14+ with pgvector extension
- Python 3.11+ or Docker
- Redis (or RabbitMQ)
- An embedding provider (Infinity Docker recommended)

## Installation

### Using uv (Recommended)
```bash
uv pip install v10r
```

### Using pip
```bash
pip install v10r
```

### Using Docker
```bash
docker pull ghcr.io/yourusername/v10r:latest
```

## Quick Start

### 1. Initialize Configuration

```bash
v10r init --database postgresql://user:pass@localhost/mydb
```

This creates `v10r-config.yaml`:

```yaml
databases:
  - name: mydb
    connection_string: postgresql://user:pass@localhost/mydb
    tables:
      - name: documents
        text_column: content
        embedding_column: content_embedding
        embedding_models:
          - infinity_bge_m3
```

### 2. Start Infinity Docker

```bash
docker run -d -p 7997:7997 \
  michaelf34/infinity:latest \
  v2 --model-id BAAI/bge-m3
```

### 3. Run Schema Reconciliation

```bash
v10r schema reconcile --config v10r-config.yaml
```

v10r automatically:
- Adds the `content_embedding` column to your table
- Creates tracking columns for model versions
- Sets up PostgreSQL triggers

### 4. Start Services

In separate terminals:

```bash
# Terminal 1: Start the listener
v10r listen --config v10r-config.yaml

# Terminal 2: Start workers (scale as needed)
v10r worker --config v10r-config.yaml --workers 4
```

### 5. Verify It's Working

```sql
-- Insert test data
INSERT INTO documents (content) VALUES ('Hello, vector world!');

-- Check embedding (appears within seconds)
SELECT content, content_embedding IS NOT NULL as has_embedding 
FROM documents;
```

## What's Next?

- **Scale Workers**: Add more workers for faster processing
- **Multiple Tables**: Add more tables to your configuration
- **Custom Models**: Use OpenAI or custom embedding APIs
- **Text Preprocessing**: Add HTML cleaning, markdown conversion
- **Production Deploy**: Use Docker Compose or Kubernetes

## Common Patterns

### Batch Import
```sql
-- v10r automatically processes all rows
INSERT INTO documents (content)
SELECT text_data FROM legacy_system;
```

### Monitoring Progress
```bash
# Check health
v10r health --config v10r-config.yaml

# View queue status
v10r queue-status --config v10r-config.yaml
```

### Using Different Models
```yaml
embedding_providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: text-embedding-3-small

tables:
  - name: products
    embedding_models:
      - openai
```

## Troubleshooting

### No Embeddings Generated?
1. Check listener is running: `v10r health`
2. Verify workers are processing: Check worker logs
3. Ensure PostgreSQL NOTIFY is enabled

### Performance Issues?
1. Scale workers: `--workers 8`
2. Use batch processing in config
3. Consider Infinity Docker over API providers

### Connection Errors?
```bash
# Test connections
v10r test-connection --config v10r-config.yaml
```

## Next Steps

- Read the [Configuration Guide](configuration.md) for advanced options
- Understand the [Architecture](architecture.md)
- Explore [API Reference](api-reference.md) for custom integrations