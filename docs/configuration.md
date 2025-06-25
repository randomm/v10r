# Configuration Reference

v10r uses YAML configuration with environment variable expansion.

## Minimal Configuration

```yaml
databases:
  - name: mydb
    connection_string: postgresql://localhost/mydb
    tables:
      - name: documents
        text_column: content
        embedding_column: content_embedding
        embedding_models:
          - infinity_bge_m3
```

## Full Configuration

```yaml
# Embedding provider settings
embedding_providers:
  infinity_bge_m3:
    type: infinity
    url: http://localhost:7997
    model: BAAI/bge-m3
    
  openai:
    type: openai
    api_key: ${OPENAI_API_KEY}
    model: text-embedding-3-small
    dimensions: 1536
    
  custom_api:
    type: custom
    url: https://api.example.com/v1
    api_key: ${CUSTOM_API_KEY}
    model: custom-model-v1

# Text preprocessing pipelines
preprocessing:
  clean_html:
    - html_to_markdown
    - normalize_whitespace
    - remove_excess_newlines
  
  minimal:
    - normalize_whitespace

# Database configurations
databases:
  - name: production
    connection_string: ${DATABASE_URL}
    ssl_mode: require
    pool_size: 20
    
    tables:
      - name: articles
        text_column: body
        embedding_column: body_vector
        embedding_models:
          - infinity_bge_m3
        preprocessing_pipeline: clean_html
        
        # Advanced options
        batch_size: 100
        update_mode: async
        
        # Metadata tracking
        track_model_version: true
        track_generated_at: true
        
      - name: products
        text_column: description
        embedding_column: description_embedding
        embedding_models:
          - openai
        preprocessing_pipeline: minimal

# Worker configuration
worker:
  batch_size: 50
  batch_timeout_seconds: 5
  max_retries: 3
  retry_delay_seconds: 60

# Queue configuration
queue:
  type: redis  # or rabbitmq
  redis:
    url: ${REDIS_URL:-redis://localhost:6379}
    max_connections: 50
  
  # Alternative: RabbitMQ
  # rabbitmq:
  #   url: amqp://guest:guest@localhost:5672/
  #   exchange: v10r_tasks
  #   queue: v10r_embeddings

# Monitoring and logging
monitoring:
  log_level: INFO
  metrics_enabled: true
  metrics_port: 9090
  
  # Health check endpoints
  health_check:
    enabled: true
    port: 8080
    path: /health

# Schema management
schema:
  # How to handle schema conflicts
  collision_strategy: auto  # auto, suffix, error
  
  # Naming conventions
  naming:
    embedding_suffix: _embedding
    model_version_suffix: _model_version
    generated_at_suffix: _generated_at
```

## Configuration Sections

### Embedding Providers

Define available embedding services:

```yaml
embedding_providers:
  provider_name:
    type: infinity|openai|custom
    # Provider-specific settings
```

**Infinity Settings**:
- `url`: Infinity server URL
- `model`: Model identifier

**OpenAI Settings**:
- `api_key`: OpenAI API key
- `model`: Model name
- `dimensions`: Embedding dimensions

**Custom Settings**:
- `url`: API endpoint
- `api_key`: Authentication key
- `model`: Model identifier
- `headers`: Additional headers

### Preprocessing

Define text cleaning pipelines:

```yaml
preprocessing:
  pipeline_name:
    - cleaner_function_1
    - cleaner_function_2
```

Available cleaners:
- `html_to_markdown`: Convert HTML to Markdown
- `normalize_whitespace`: Clean whitespace
- `remove_excess_newlines`: Limit consecutive newlines
- `lowercase`: Convert to lowercase
- `remove_punctuation`: Strip punctuation

### Database Configuration

```yaml
databases:
  - name: identifier
    connection_string: postgresql://...
    ssl_mode: disable|require|verify-ca|verify-full
    pool_size: 20
    tables:
      # Table configurations
```

### Table Configuration

```yaml
tables:
  - name: table_name
    text_column: source_column
    embedding_column: target_column
    embedding_models:
      - model_name
    preprocessing_pipeline: pipeline_name
    
    # Optional settings
    batch_size: 100
    update_mode: async|sync
    track_model_version: true|false
    track_generated_at: true|false
    
    # Column naming overrides
    model_version_column: custom_name
    generated_at_column: custom_name
```

### Worker Settings

```yaml
worker:
  batch_size: 50              # Rows per batch
  batch_timeout_seconds: 5    # Max wait for full batch
  max_retries: 3             # Retry failed embeddings
  retry_delay_seconds: 60    # Delay between retries
  concurrent_batches: 4      # Parallel batch processing
```

### Queue Configuration

**Redis** (recommended):
```yaml
queue:
  type: redis
  redis:
    url: redis://localhost:6379
    max_connections: 50
    key_prefix: v10r
```

**RabbitMQ**:
```yaml
queue:
  type: rabbitmq
  rabbitmq:
    url: amqp://localhost:5672/
    exchange: v10r_tasks
    queue: v10r_embeddings
    durable: true
```

## Environment Variables

Use `${VAR_NAME}` syntax for environment variables:

```yaml
connection_string: ${DATABASE_URL}
api_key: ${OPENAI_API_KEY:-default_value}
```

## Best Practices

### Security
- Use environment variables for credentials
- Enable SSL for database connections
- Rotate API keys regularly

### Performance
- Set appropriate batch sizes (50-200)
- Use connection pooling
- Scale workers based on load

### Reliability
- Enable retries for transient failures
- Monitor queue depth
- Set up health checks

## Configuration Validation

Validate your configuration:

```bash
v10r validate-config --config v10r-config.yaml
```

Common validation errors:
- Missing required fields
- Invalid connection strings
- Unknown embedding providers
- Circular preprocessing pipelines