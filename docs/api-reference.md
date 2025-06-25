# API Reference

## CLI Commands

### Core Commands

#### `v10r init`
Initialize a new configuration file.

```bash
v10r init --database postgresql://localhost/mydb [--output config.yaml]
```

Options:
- `--database`: PostgreSQL connection string (required)
- `--output`: Configuration file path (default: v10r-config.yaml)

#### `v10r listen`
Start the listener service.

```bash
v10r listen --config v10r-config.yaml [--workers-in-process 0]
```

Options:
- `--config`: Configuration file path
- `--workers-in-process`: Run workers in same process (development)

#### `v10r worker`
Start worker processes.

```bash
v10r worker --config v10r-config.yaml [--workers 4]
```

Options:
- `--config`: Configuration file path
- `--workers`: Number of worker processes (default: CPU count)

### Management Commands

#### `v10r validate-config`
Validate configuration file syntax and settings.

```bash
v10r validate-config --config v10r-config.yaml
```

#### `v10r schema reconcile`
Update database schema to match configuration.

```bash
v10r schema reconcile --config v10r-config.yaml [--dry-run]
```

Options:
- `--dry-run`: Show changes without applying

#### `v10r register-table`
Register a new table for vectorization.

```bash
v10r register-table --config v10r-config.yaml \
  --database mydb --table products \
  --text-column description \
  --embedding-model infinity_bge_m3
```

### Utility Commands

#### `v10r health`
Check system health.

```bash
v10r health --config v10r-config.yaml
```

Returns:
- Database connectivity
- Queue status
- Embedding provider status
- Worker status

#### `v10r test-connection`
Test all configured connections.

```bash
v10r test-connection --config v10r-config.yaml
```

## Core Python APIs

### EmbeddingClient Base Class

All embedding providers implement this interface:

```python
from v10r.embedding.base import EmbeddingClient

class EmbeddingClient(ABC):
    @abstractmethod
    async def embed_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings for text batch."""
        
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
```

### Custom Embedding Provider

Create custom providers by extending the base class:

```python
from v10r.embedding.base import EmbeddingClient

class MyEmbeddingClient(EmbeddingClient):
    def __init__(self, config: dict):
        self.api_url = config["api_url"]
        self.model = config["model"]
    
    async def embed_batch(
        self, texts: List[str]
    ) -> List[List[float]]:
        # Your implementation here
        pass
```

### Text Preprocessing

Implement custom text cleaners:

```python
from v10r.preprocessing.cleaners import TextCleaner

class MyCustomCleaner(TextCleaner):
    def clean(self, text: str) -> str:
        # Your preprocessing logic
        return processed_text
```

### Schema Operations

Programmatic schema management:

```python
from v10r.schema import SchemaReconciler
from v10r.config import Config

config = Config.from_file("v10r-config.yaml")
reconciler = SchemaReconciler(config)

# Check what changes would be made
changes = await reconciler.plan_changes()

# Apply changes
await reconciler.reconcile()
```

## Extension Points

### Custom Queue Backends

Implement the queue interface:

```python
from v10r.queue.base import QueueBackend

class CustomQueue(QueueBackend):
    async def push(self, task: dict) -> None:
        """Add task to queue."""
        
    async def pop(self) -> Optional[dict]:
        """Get next task."""
```

### Custom Collision Handlers

Handle schema naming conflicts:

```python
from v10r.schema.collision_detector import CollisionHandler

class MyCollisionHandler(CollisionHandler):
    def resolve_column_name(
        self, 
        base_name: str, 
        existing_columns: List[str]
    ) -> str:
        # Your resolution logic
        return new_name
```

### Worker Lifecycle Hooks

Add custom processing logic:

```python
from v10r.worker import Worker

class CustomWorker(Worker):
    async def pre_process(self, task: dict) -> dict:
        """Called before embedding generation."""
        return task
        
    async def post_process(
        self, task: dict, embedding: List[float]
    ) -> None:
        """Called after embedding generation."""
        pass
```

## Events and Hooks

### PostgreSQL Functions

v10r creates these database functions:

```sql
-- Notification function
CREATE FUNCTION v10r_notify() RETURNS trigger

-- Metadata tracking
CREATE FUNCTION v10r_update_metadata() RETURNS trigger
```

### NOTIFY Payload Format

```json
{
    "table": "documents",
    "id": 123,
    "operation": "INSERT",
    "timestamp": "2024-01-01T12:00:00Z"
}
```

### Queue Task Format

```json
{
    "database": "mydb",
    "table": "documents",
    "id": 123,
    "text_column": "content",
    "embedding_column": "content_embedding",
    "embedding_model": "infinity_bge_m3",
    "preprocessing": ["html_to_markdown", "normalize"]
}
```