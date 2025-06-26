"""
Configuration system for v10r using Pydantic.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError, validator, ConfigDict
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError


class DatabaseConnection(BaseModel):
    """Database connection configuration."""

    host: str = Field(..., description="Database host")
    port: int = Field(5432, description="Database port")
    database: str = Field(..., description="Database name")
    user: str = Field(..., description="Database user")
    password: str = Field(..., description="Database password")
    ssl_mode: str = Field("prefer", description="SSL mode")
    connect_timeout: int = Field(30, description="Connection timeout in seconds")
    command_timeout: int = Field(60, description="Command timeout in seconds")

    def to_dsn(self) -> str:
        """Convert to PostgreSQL DSN string."""
        return (
            f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/"
            f"{self.database}?sslmode={self.ssl_mode}"
        )


class PreprocessingConfig(BaseModel):
    """Text preprocessing configuration."""

    enabled: bool = Field(False, description="Enable text preprocessing")
    cleaned_text_column: Optional[str] = Field(
        None, description="Column to store cleaned text"
    )
    cleaning_config: str = Field(
        "basic_cleanup", description="Cleaning configuration name"
    )


class TableConfig(BaseModel):
    """Configuration for a single table to vectorize."""

    schema: str = Field("public", description="Database schema")
    table: str = Field(..., description="Table name")
    text_column: str = Field(..., description="Column containing text to vectorize")
    id_column: str = Field("id", description="Primary key column")
    vector_column: Optional[str] = Field(
        None, description="Column to store vector embeddings"
    )
    model_column: Optional[str] = Field(
        None, description="Column to store model information"
    )
    embedding_config: str = Field(..., description="Embedding configuration name")
    preprocessing: Optional[PreprocessingConfig] = Field(
        None, description="Text preprocessing configuration"
    )
    ensure_no_nulls: bool = Field(
        True, description="Ensure no NULL vectors remain"
    )
    vectorize_conditions: Optional[List[str]] = Field(
        None, description="SQL conditions for when to vectorize"
    )

    @validator("vector_column", always=True)
    def set_vector_column(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if v is None:
            text_col = values.get("text_column", "content")
            return f"{text_col}_vector"
        return v

    @validator("model_column", always=True)
    def set_model_column(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if v is None:
            vector_col = values.get("vector_column")
            if vector_col:
                return f"{vector_col}_model"
            text_col = values.get("text_column", "content")
            return f"{text_col}_embedding_model"
        return v

    @property
    def full_table_name(self) -> str:
        """Get the full table name with schema."""
        return f"{self.schema}.{self.table}"


class DatabaseConfig(BaseModel):
    """Configuration for a single database."""

    name: str = Field(..., description="Database configuration name")
    connection: DatabaseConnection = Field(..., description="Connection details")
    tables: List[TableConfig] = Field(
        default_factory=list, description="Tables to vectorize"
    )


class EmbeddingConfig(BaseModel):
    """Configuration for an embedding provider."""

    provider: Literal["openai", "azure_openai", "infinity", "custom"] = Field(
        ..., description="Embedding provider type"
    )
    api_key: Optional[str] = Field(None, description="API key for the provider")
    endpoint: Optional[str] = Field(None, description="Custom endpoint URL")
    model: str = Field(..., description="Model name")
    dimensions: int = Field(..., description="Vector dimensions")
    batch_size: int = Field(100, description="Batch size for embedding requests")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    retry_delay: float = Field(1.0, description="Base retry delay in seconds")


class QueueConfig(BaseModel):
    """Queue configuration."""

    type: Literal["redis", "rabbitmq"] = Field("redis", description="Queue type")
    connection: Dict[str, Any] = Field(
        default_factory=dict, description="Queue connection parameters"
    )
    name: str = Field("v10r_tasks", description="Queue name")
    max_retries: int = Field(3, description="Maximum task retries")
    retry_delay: int = Field(60, description="Retry delay in seconds")


class WorkerConfig(BaseModel):
    """Worker configuration."""

    concurrency: int = Field(4, description="Number of concurrent workers")
    batch_timeout: int = Field(60, description="Batch timeout in seconds")
    max_batch_size: int = Field(100, description="Maximum batch size")
    heartbeat_interval: int = Field(30, description="Heartbeat interval in seconds")


class SchemaManagementConfig(BaseModel):
    """Schema management configuration."""

    reconciliation: bool = Field(True, description="Enable schema reconciliation")
    mode: Literal["safe", "force", "dry_run"] = Field(
        "safe", description="Reconciliation mode"
    )
    check_interval: int = Field(300, description="Check interval in seconds")
    collision_strategy: Literal["interactive", "auto_warn", "auto_silent", "error"] = (
        Field("auto_warn", description="Column collision handling strategy")
    )
    allow_drop_columns: bool = Field(
        False, description="Allow dropping columns during reconciliation"
    )
    require_backup: bool = Field(
        True, description="Require backup before schema changes"
    )


class CleaningPipelineStep(BaseModel):
    """A single step in a text cleaning pipeline."""

    type: str = Field(..., description="Step type")
    method: Optional[str] = Field(None, description="Method to use")
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Step options"
    )


class CleaningConfig(BaseModel):
    """Text cleaning configuration."""

    name: str = Field(..., description="Cleaning configuration name")
    pipeline: List[CleaningPipelineStep] = Field(
        ..., description="Cleaning pipeline steps"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO", description="Log level"
    )
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Optional[str] = Field(None, description="Log file path")
    max_size: int = Field(10485760, description="Max log file size in bytes")  # 10MB
    backup_count: int = Field(5, description="Number of backup log files")


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enabled: bool = Field(True, description="Enable monitoring")
    metrics_port: int = Field(9090, description="Metrics server port")
    health_check_port: int = Field(8080, description="Health check port")
    prometheus_enabled: bool = Field(True, description="Enable Prometheus metrics")
    jaeger_enabled: bool = Field(False, description="Enable Jaeger tracing")
    jaeger_endpoint: Optional[str] = Field(None, description="Jaeger endpoint")


class V10rConfig(BaseSettings):
    """Main v10r configuration."""

    # Service configuration
    service_name: str = Field("v10r", description="Service name")
    debug: bool = Field(False, description="Enable debug mode")
    dry_run: bool = Field(False, description="Enable dry run mode")

    # Core components
    databases: List[DatabaseConfig] = Field(
        default_factory=list, description="Database configurations"
    )
    embeddings: Dict[str, EmbeddingConfig] = Field(
        default_factory=dict, description="Embedding configurations"
    )
    queue: QueueConfig = Field(
        default_factory=QueueConfig, description="Queue configuration"
    )
    workers: WorkerConfig = Field(
        default_factory=WorkerConfig, description="Worker configuration"
    )

    # Advanced features
    schema_management: SchemaManagementConfig = Field(
        default_factory=SchemaManagementConfig,
        description="Schema management configuration",
    )
    cleaning_configs: Dict[str, CleaningConfig] = Field(
        default_factory=dict, description="Text cleaning configurations"
    )

    # System configuration
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig, description="Monitoring configuration"
    )

    model_config = ConfigDict(
        env_file=".env",
        env_prefix="V10R_",
        case_sensitive=False,
        extra="ignore"
    )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "V10rConfig":
        """Load configuration from a YAML file."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Expand environment variables in the data
            data = cls._expand_env_vars(data)

            return cls(**data)
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except ValidationError as e:
            raise ConfigurationError(f"Invalid configuration: {e}")

    @classmethod
    def _expand_env_vars(cls, data: Any) -> Any:
        """Recursively expand environment variables in configuration data."""
        if isinstance(data, dict):
            return {k: cls._expand_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls._expand_env_vars(item) for item in data]
        elif isinstance(data, str):
            return os.path.expandvars(data)
        else:
            return data

    def get_database(self, name: str) -> DatabaseConfig:
        """Get database configuration by name."""
        for db in self.databases:
            if db.name == name:
                return db
        raise ConfigurationError(f"Database configuration '{name}' not found")

    def get_embedding_config(self, name: str) -> EmbeddingConfig:
        """Get embedding configuration by name."""
        if name not in self.embeddings:
            raise ConfigurationError(f"Embedding configuration '{name}' not found")
        return self.embeddings[name]

    def get_cleaning_config(self, name: str) -> CleaningConfig:
        """Get cleaning configuration by name."""
        if name not in self.cleaning_configs:
            raise ConfigurationError(f"Cleaning configuration '{name}' not found")
        return self.cleaning_configs[name]

    def validate_config(self) -> None:
        """Validate the entire configuration for consistency."""
        # Check that all referenced embedding configs exist
        for db in self.databases:
            for table in db.tables:
                if table.embedding_config not in self.embeddings:
                    raise ConfigurationError(
                        f"Table {table.full_table_name} references unknown "
                        f"embedding config '{table.embedding_config}'"
                    )

                # Check preprocessing references
                if table.preprocessing and table.preprocessing.enabled:
                    cleaning_name = table.preprocessing.cleaning_config
                    if cleaning_name not in self.cleaning_configs:
                        raise ConfigurationError(
                            f"Table {table.full_table_name} references unknown "
                            f"cleaning config '{cleaning_name}'"
                        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.dict(exclude_none=True), f, default_flow_style=False, indent=2
            ) 