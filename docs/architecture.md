# v10r Architecture

## System Overview

v10r uses PostgreSQL's NOTIFY/LISTEN mechanism to create a real-time, event-driven vectorization pipeline.

```
┌─────────────┐      NOTIFY      ┌──────────┐      Queue      ┌────────┐
│ PostgreSQL  │ ───────────────▶ │ Listener │ ─────────────▶ │ Worker │
│   Tables    │                  │ Service  │                 │  Pool  │
└─────────────┘                  └──────────┘                 └────────┘
      ▲                                                            │
      │                                                            │
      └────────────────── UPDATE with embeddings ──────────────────┘
```

## Core Components

### 1. Database Layer

**Trigger System**: PostgreSQL triggers monitor INSERT/UPDATE operations and send NOTIFY events with row identifiers.

**Schema Management**: The `SchemaReconciler` ensures database schema matches configuration:
- Adds embedding columns with proper types
- Creates metadata tracking columns
- Manages naming collisions intelligently

### 2. Listener Service

**Event Processing**: Single process monitoring PostgreSQL NOTIFY channels:
- Receives change notifications
- Deduplicates events (100ms window)
- Queues tasks for worker processing

**Queue Interface**: Supports Redis (default) and RabbitMQ through abstract interface.

### 3. Worker Pool

**Parallel Processing**: Multiple workers consume queue tasks:
- Fetches row data from PostgreSQL
- Generates embeddings via configured provider
- Updates database with results

**Batch Optimization**: Groups multiple rows for efficient API calls.

### 4. Embedding Providers

**Provider Abstraction**: Clean interface supporting:
- **Infinity Docker**: Local BGE-M3 model (recommended)
- **OpenAI API**: Cloud-based embeddings
- **Custom APIs**: Any OpenAI-compatible endpoint

## Data Flow

### 1. Change Detection
```sql
-- Trigger on INSERT/UPDATE
CREATE TRIGGER documents_notify_v10r
AFTER INSERT OR UPDATE ON documents
FOR EACH ROW EXECUTE FUNCTION v10r_notify();
```

### 2. Event Notification
```python
# PostgreSQL sends NOTIFY with payload
NOTIFY v10r_documents, '{"id": 123, "table": "documents"}'
```

### 3. Queue Task
```python
# Listener queues embedding task
{
    "database": "mydb",
    "table": "documents", 
    "id": 123,
    "text_column": "content",
    "embedding_column": "content_embedding"
}
```

### 4. Embedding Generation
```python
# Worker processes task
1. Fetch text from database
2. Preprocess text (if configured)
3. Generate embedding via provider
4. Update database with results
```

## Key Design Decisions

### Event-Driven Architecture
- **Why**: Enables real-time processing without polling
- **Benefit**: Minimal latency, efficient resource usage

### Schema Reconciliation
- **Why**: Eliminates manual schema management
- **Benefit**: True "set it and forget it" operation

### Provider Abstraction
- **Why**: Flexibility in embedding sources
- **Benefit**: Easy migration between providers

### Stateless Workers
- **Why**: Horizontal scalability
- **Benefit**: Add/remove workers dynamically

## Deployment Patterns

### Development
```bash
# Single process mode
v10r listen --workers-in-process 2
```

### Production
```yaml
# Docker Compose
services:
  listener:
    image: v10r:latest
    command: listen
  
  worker:
    image: v10r:latest
    command: worker --workers 4
    deploy:
      replicas: 3
```

### High Availability
- Run multiple workers across machines
- Use Redis Sentinel or cluster
- PostgreSQL streaming replication

## Performance Considerations

### Throughput
- **Bottleneck**: Usually embedding API rate limits
- **Solution**: Use Infinity Docker for unlimited local processing

### Latency
- **NOTIFY**: ~1-10ms detection
- **Queue**: ~1ms Redis latency
- **Embedding**: 50-500ms depending on provider

### Scaling
- **Workers**: Linear scaling with worker count
- **Queue**: Redis handles 100k+ ops/second
- **Database**: PostgreSQL connection pooling

## Security Model

### Database
- Separate read/write credentials
- SSL/TLS connections required
- No credential storage in database

### API Keys
- Environment variable expansion
- Never logged or stored
- Encrypted in transit

### Network
- Internal services on private networks
- TLS for external API calls
- No public endpoints required