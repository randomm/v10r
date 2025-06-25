# v10r Documentation

Welcome to v10r - the "set it and forget it" PostgreSQL vectorization service.

## What is v10r?

v10r automatically creates and maintains vector embeddings for text data in PostgreSQL databases. Point it at any table containing text, and v10r handles everything: schema management, real-time updates, and embedding generation.

## Quick Links

- [Getting Started](getting-started.md) - Setup and first vectorization
- [Configuration Guide](configuration.md) - YAML configuration reference
- [Architecture Overview](architecture.md) - System design and components
- [API Reference](api-reference.md) - Core APIs and interfaces

## Key Features

- **Zero Configuration**: Automatic schema detection and management
- **Real-Time Processing**: PostgreSQL NOTIFY-based event system
- **Multiple Providers**: Infinity Docker, OpenAI, and custom APIs
- **Production Ready**: Docker deployment, health checks, monitoring
- **Schema Safety**: Intelligent collision detection and resolution

## Navigation

### User Guide
- [Getting Started](getting-started.md) - Installation and quick start
- [Configuration](configuration.md) - Complete configuration reference
- [CLI Commands](api-reference.md#cli-commands) - Command-line interface

### Developer Guide
- [Development Setup](development.md) - Local development environment
- [Architecture](architecture.md) - System design and flow
- [API Reference](api-reference.md) - Core APIs and extensions
- [Contributing](../CONTRIBUTING.md) - How to contribute

### Examples
- [Simple Configuration](../examples/simple-config.yaml) - Basic setup
- [Full Configuration](../examples/v10r-config.yaml) - All options

## Support

- **Issues**: [GitHub Issues](https://github.com/randomm/v10r/issues)
- **Discussions**: [GitHub Discussions](https://github.com/randomm/v10r/discussions)
- **Contributing**: See [CONTRIBUTING.md](../CONTRIBUTING.md)