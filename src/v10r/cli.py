"""
Command-line interface for v10r.
"""

import asyncio
import sys
from functools import wraps
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import V10rConfig
from .exceptions import ConfigurationError, V10rError


console = Console()


def handle_errors(func):
    """Decorator to handle errors gracefully in CLI commands."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except V10rError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"[red]Unexpected error:[/red] {e}")
            if "--debug" in sys.argv:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    return wrapper


@click.group()
@click.version_option(__version__)
@click.option(
    "--debug", is_flag=True, help="Enable debug mode"
)
@click.pass_context
def main(ctx, debug):
    """v10r: Generic PostgreSQL NOTIFY-based vectorization service."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@main.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="v10r-config.yaml",
    help="Output configuration file path",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    help="Interactive configuration wizard",
)
@handle_errors
def init(output: str, interactive: bool):
    """Initialize a new v10r configuration file."""
    if Path(output).exists():
        if not click.confirm(f"Configuration file {output} already exists. Overwrite?"):
            return

    if interactive:
        config = _interactive_config_wizard()
    else:
        config = _create_default_config()

    config.to_yaml(output)
    console.print(f"[green]✓[/green] Configuration file created: {output}")
    console.print("\n[yellow]Next steps:[/yellow]")
    console.print("1. Edit the configuration file with your database and API details")
    console.print("2. Set up your PostgreSQL database with pgvector extension")
    console.print("3. Run: v10r validate-config")
    console.print("4. Run: v10r listen --config your-config.yaml")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@handle_errors
def validate_config(config: str):
    """Validate configuration file."""
    console.print(f"Validating configuration: {config}")
    
    try:
        v10r_config = V10rConfig.from_yaml(config)
        v10r_config.validate_config()
        
        console.print("[green]✓[/green] Configuration is valid")
        
        # Display configuration summary
        _display_config_summary(v10r_config)
        
    except ConfigurationError as e:
        console.print(f"[red]✗[/red] Configuration error: {e}")
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@click.option(
    "--channel",
    default="v10r_events",
    help="PostgreSQL NOTIFY channel to listen on",
)
@handle_errors
def listen(config: str, channel: str):
    """Start the v10r listener service."""
    console.print(f"[blue]Starting v10r listener service...[/blue]")
    console.print(f"Config: {config}")
    console.print(f"Channel: {channel}")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    # Import here to avoid circular imports
    from .listener import VectorListener
    
    async def run_listener():
        listener = VectorListener(v10r_config, channel)
        await listener.start()
    
    try:
        asyncio.run(run_listener())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down listener...[/yellow]")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    help="Number of worker processes (overrides config)",
)
@handle_errors
def worker(config: str, workers: Optional[int]):
    """Start v10r worker processes."""
    console.print(f"[blue]Starting v10r worker service...[/blue]")
    console.print(f"Config: {config}")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    if workers:
        v10r_config.workers.concurrency = workers
        console.print(f"Workers: {workers} (overridden)")
    else:
        console.print(f"Workers: {v10r_config.workers.concurrency}")
    
    # Import here to avoid circular imports
    from .worker import VectorizerWorker
    
    async def run_workers():
        worker = VectorizerWorker(v10r_config)
        await worker.start()
    
    try:
        asyncio.run(run_workers())
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down workers...[/yellow]")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@handle_errors
def status(config: str):
    """Show service status and statistics."""
    console.print(f"[blue]v10r Service Status[/blue]")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    async def run_status_check():
        from .database.connection import DatabaseManager
        import psutil
        import redis
        import time
        
        # Configuration summary
        console.print(f"\n[bold cyan]Configuration[/bold cyan]")
        console.print(f"  Databases: {len(v10r_config.databases)}")
        console.print(f"  Embedding Configs: {len(v10r_config.embeddings)}")
        
        total_tables = sum(len(db.tables) for db in v10r_config.databases)
        console.print(f"  Tables to Monitor: {total_tables}")
        
        # Database status
        console.print(f"\n[bold cyan]Database Status[/bold cyan]")
        
        for db_config in v10r_config.databases:
            console.print(f"\n  📊 Database: [yellow]{db_config.name}[/yellow]")
            
            try:
                db_manager = DatabaseManager(db_config)
                await db_manager.initialize()
                
                async with db_manager.get_connection() as conn:
                    # Check connection
                    start_time = time.time()
                    await conn.fetchval("SELECT 1")
                    response_time = (time.time() - start_time) * 1000
                    
                    # Get database stats
                    stats = await conn.fetchrow("""
                        SELECT 
                            count(*) as active_connections,
                            current_database() as database_name
                        FROM pg_stat_activity 
                        WHERE state = 'active'
                    """)
                    
                console.print(f"    ✅ [green]Connected[/green] ({response_time:.1f}ms)")
                console.print(f"    Database: {stats['database_name']}")
                console.print(f"    Active Connections: {stats['active_connections']}")
                
                # Check table status
                for table_config in db_config.tables:
                    table_name = f"{table_config.schema}.{table_config.table}"
                    try:
                        async with db_manager.get_connection() as conn:
                            # Check if table exists
                            exists = await conn.fetchval("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_schema = $1 AND table_name = $2
                                )
                            """, table_config.schema, table_config.table)
                            
                            if exists:
                                # Get row count
                                row_count = await conn.fetchval(f"""
                                    SELECT count(*) FROM {table_name}
                                """)
                                
                                # Check if vector column exists
                                vector_col_exists = await conn.fetchval("""
                                    SELECT EXISTS (
                                        SELECT FROM information_schema.columns 
                                        WHERE table_schema = $1 AND table_name = $2 
                                        AND column_name LIKE '%_vector'
                                    )
                                """, table_config.schema, table_config.table)
                                
                                status_icon = "🔄" if vector_col_exists else "⏸️"
                                vector_status = "Configured" if vector_col_exists else "Not configured"
                                
                                console.print(f"    {status_icon} Table {table_name}: {row_count:,} rows ({vector_status})")
                            else:
                                console.print(f"    ❌ Table {table_name}: Not found")
                                
                    except Exception as e:
                        console.print(f"    ⚠️  Table {table_name}: Error - {e}")
                
                await db_manager.close()
                
            except Exception as e:
                console.print(f"    ❌ [red]Connection failed: {e}[/red]")
        
        # Redis/Queue status (if configured)
        console.print(f"\n[bold cyan]Queue Status[/bold cyan]")
        try:
            # Try to connect to Redis (assuming default Redis settings)
            import os
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', '6379'))
            
            redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            redis_info = redis_client.info()
            
            console.print(f"  ✅ [green]Redis connected[/green]")
            console.print(f"    Version: {redis_info.get('redis_version', 'Unknown')}")
            console.print(f"    Connected clients: {redis_info.get('connected_clients', 0)}")
            console.print(f"    Used memory: {redis_info.get('used_memory_human', 'Unknown')}")
            
            # Check queue lengths
            queue_length = redis_client.llen('v10r:embedding_queue')
            console.print(f"    Queue length: {queue_length}")
            
        except Exception as e:
            console.print(f"  ⚠️  [yellow]Redis not accessible: {e}[/yellow]")
        
        # System resources
        console.print(f"\n[bold cyan]System Resources[/bold cyan]")
        console.print(f"  CPU Usage: {psutil.cpu_percent(interval=1):.1f}%")
        
        memory = psutil.virtual_memory()
        console.print(f"  Memory: {memory.percent:.1f}% used ({memory.used // (1024**3):.1f}GB / {memory.total // (1024**3):.1f}GB)")
        
        disk = psutil.disk_usage('/')
        console.print(f"  Disk: {disk.percent:.1f}% used ({disk.used // (1024**3):.1f}GB / {disk.total // (1024**3):.1f}GB)")
        
        # Process information (look for v10r processes)
        console.print(f"\n[bold cyan]v10r Processes[/bold cyan]")
        v10r_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'v10r' in cmdline and ('listen' in cmdline or 'worker' in cmdline):
                    v10r_processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if v10r_processes:
            for proc in v10r_processes:
                cmdline = ' '.join(proc['cmdline'])
                memory_mb = proc['memory_info'].rss / (1024 * 1024)
                console.print(f"  🔄 PID {proc['pid']}: {memory_mb:.1f}MB")
                console.print(f"    Command: {cmdline}")
        else:
            console.print(f"  ⏸️  [yellow]No v10r processes running[/yellow]")
        
        console.print(f"\n✨ [bold]Status check complete[/bold]")
    
    try:
        asyncio.run(run_status_check())
    except KeyboardInterrupt:
        console.print("\n[yellow]Status check cancelled[/yellow]")
        exit(1)
    except Exception as e:
        console.print(f"\n[red]Status check error: {e}[/red]")
        exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@click.option(
    "--database",
    required=True,
    help="Database name",
)
@click.option(
    "--schema",
    default="public",
    help="Database schema",
)
@click.option(
    "--table",
    required=True,
    help="Table name",
)
@click.option(
    "--text-column",
    required=True,
    help="Text column to vectorize",
)
@click.option(
    "--vector-column",
    help="Vector column name (auto-generated if not specified)",
)
@click.option(
    "--model-column",
    help="Model tracking column name (auto-generated if not specified)",
)
@click.option(
    "--embedding-config",
    required=True,
    help="Embedding configuration name",
)
@click.option(
    "--collision-strategy",
    type=click.Choice(["interactive", "auto_warn", "auto_silent", "error"]),
    default="interactive",
    help="Column collision handling strategy",
)
@handle_errors
def register(
    config: str,
    database: str,
    schema: str,
    table: str,
    text_column: str,
    vector_column: Optional[str],
    model_column: Optional[str],
    embedding_config: str,
    collision_strategy: str,
):
    """Register a table for vectorization."""
    console.print(f"[blue]Registering table for vectorization...[/blue]")
    console.print(f"Database: {database}")
    console.print(f"Table: {schema}.{table}")
    console.print(f"Text Column: {text_column}")
    console.print(f"Embedding Config: {embedding_config}")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    # Import here to avoid circular imports
    from .registration import TableRegistrationService
    
    async def run_registration():
        service = TableRegistrationService(v10r_config)
        result = await service.register_table(
            database=database,
            schema=schema,
            table=table,
            text_column=text_column,
            vector_column=vector_column,
            model_column=model_column,
            embedding_config=embedding_config,
            collision_strategy=collision_strategy,
        )
        
        if result.success:
            console.print(f"[green]✓ Table {schema}.{table} registered successfully![/green]")
            console.print(f"Vector column: {result.vector_column}")
            console.print(f"Model column: {result.model_column}")
            if result.timestamp_column:
                console.print(f"Timestamp column: {result.timestamp_column}")
            
            if result.warnings:
                console.print("\n[yellow]Warnings:[/yellow]")
                for warning in result.warnings:
                    console.print(f"  • {warning}")
        else:
            console.print(f"[red]✗ Registration failed: {result.error}[/red]")
            return 1
        
        return 0
    
    try:
        exit_code = asyncio.run(run_registration())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Registration cancelled[/yellow]")
        exit(1)
    except Exception as e:
        console.print(f"\n[red]Registration error: {e}[/red]")
        exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without making changes",
)
@handle_errors
def schema_reconcile(config: str, dry_run: bool):
    """Reconcile database schema with configuration."""
    console.print(f"[blue]Schema reconciliation[/blue]")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    if dry_run:
        console.print("[yellow]Dry run mode - no changes will be made[/yellow]")
    
    # TODO: Implement schema reconciliation
    console.print("Schema reconciliation not yet implemented")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@handle_errors
def schema_status(config: str):
    """Show schema status and drift detection."""
    console.print(f"[blue]Schema Status[/blue]")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    # TODO: Implement schema status checking
    console.print("Schema status checking not yet implemented")


@main.command()
@click.option(
    "--config",
    "-c", 
    type=click.Path(exists=True),
    required=True,
    help="Configuration file path",
)
@handle_errors
def test_connection(config: str):
    """Test database and API connections."""
    console.print(f"[blue]Testing connections...[/blue]")
    
    v10r_config = V10rConfig.from_yaml(config)
    
    async def run_connection_tests():
        from .database.connection import DatabaseManager
        from .embedding.factory import EmbeddingClientFactory
        import time
        
        total_passed = 0
        total_failed = 0
        
        # Test database connections
        console.print(f"\n[bold cyan]Database Connections[/bold cyan]")
        
        for db_config in v10r_config.databases:
            console.print(f"\nTesting database: [yellow]{db_config.name}[/yellow]")
            
            try:
                db_manager = DatabaseManager(db_config)
                start_time = time.time()
                
                # Test connection
                await db_manager.initialize()
                
                # Test basic query
                async with db_manager.get_connection() as conn:
                    result = await conn.fetchval("SELECT version()")
                    
                await db_manager.close()
                
                response_time = (time.time() - start_time) * 1000
                console.print(f"  ✅ [green]Connected successfully[/green] ({response_time:.1f}ms)")
                console.print(f"     PostgreSQL version: {result.split(',')[0]}")
                total_passed += 1
                
            except Exception as e:
                console.print(f"  ❌ [red]Connection failed: {e}[/red]")
                total_failed += 1
        
        # Test embedding API connections
        console.print(f"\n[bold cyan]Embedding API Connections[/bold cyan]")
        
        for name, embedding_config in v10r_config.embeddings.items():
            console.print(f"\nTesting embedding API: [yellow]{name}[/yellow]")
            
            try:
                start_time = time.time()
                client = EmbeddingClientFactory.create_client(embedding_config)
                
                # Test health check
                health_status = await client.health_check()
                response_time = (time.time() - start_time) * 1000
                
                if health_status.get("status") == "healthy":
                    console.print(f"  ✅ [green]API healthy[/green] ({response_time:.1f}ms)")
                    console.print(f"     Provider: {embedding_config.provider}")
                    console.print(f"     Model: {embedding_config.model}")
                    console.print(f"     Dimensions: {embedding_config.dimensions}")
                    total_passed += 1
                else:
                    console.print(f"  ⚠️  [yellow]API accessible but unhealthy[/yellow]")
                    console.print(f"     Error: {health_status.get('error', 'Unknown')}")
                    total_failed += 1
                    
                # Clean up
                if hasattr(client, 'close'):
                    await client.close()
                    
            except Exception as e:
                console.print(f"  ❌ [red]API test failed: {e}[/red]")
                total_failed += 1
        
        # Summary
        console.print(f"\n[bold]Connection Test Summary[/bold]")
        console.print(f"  Passed: [green]{total_passed}[/green]")
        console.print(f"  Failed: [red]{total_failed}[/red]")
        
        if total_failed == 0:
            console.print(f"\n🎉 [bold green]All connections working![/bold green]")
            return 0
        else:
            console.print(f"\n⚠️  [bold yellow]Some connections failed[/bold yellow]")
            return 1
    
    try:
        exit_code = asyncio.run(run_connection_tests())
        exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Connection tests cancelled[/yellow]")
        exit(1)
    except Exception as e:
        console.print(f"\n[red]Connection test error: {e}[/red]")
        exit(1)


def _create_default_config() -> V10rConfig:
    """Create a default configuration with examples."""
    from .config import (
        DatabaseConfig,
        DatabaseConnection,
        EmbeddingConfig,
        TableConfig,
        CleaningConfig,
        CleaningPipelineStep,
    )
    
    # Create default embedding configurations
    embeddings = {
        "openai_small": EmbeddingConfig(
            provider="openai",
            api_key="${OPENAI_API_KEY}",
            model="text-embedding-3-small",
            dimensions=1536,
            batch_size=100,
        ),
        "infinity_static": EmbeddingConfig(
            provider="infinity",
            endpoint="${INFINITY_ENDPOINT}",
            model="minishlab/potion-multilingual-128M",
            dimensions=768,
            batch_size=50,
        ),
    }
    
    # Create default cleaning configurations
    cleaning_configs = {
        "html_to_markdown": CleaningConfig(
            name="html_to_markdown",
            pipeline=[
                CleaningPipelineStep(
                    type="html_extraction",
                    method="trafilatura",
                    options={"include_tables": True, "include_links": False},
                ),
                CleaningPipelineStep(
                    type="markdown_conversion",
                    method="markdownify",
                ),
                CleaningPipelineStep(
                    type="text_normalization",
                    method="ftfy",
                ),
                CleaningPipelineStep(
                    type="whitespace_cleanup",
                    options={"preserve_paragraphs": True},
                ),
            ],
        ),
        "basic_cleanup": CleaningConfig(
            name="basic_cleanup",
            pipeline=[
                CleaningPipelineStep(
                    type="html_strip",
                    method="beautifulsoup",
                ),
                CleaningPipelineStep(
                    type="unicode_normalize",
                    method="unidecode",
                ),
                CleaningPipelineStep(
                    type="whitespace_cleanup",
                ),
            ],
        ),
    }
    
    # Create example database configuration
    databases = [
        DatabaseConfig(
            name="primary",
            connection=DatabaseConnection(
                host="${POSTGRES_HOST}",
                port=5432,
                database="${POSTGRES_DB}",
                user="${POSTGRES_USER}",
                password="${POSTGRES_PASSWORD}",
            ),
            tables=[
                TableConfig(
                    schema="public",
                    table="documents",
                    text_column="content",
                    embedding_config="openai_small",
                ),
            ],
        )
    ]
    
    return V10rConfig(
        databases=databases,
        embeddings=embeddings,
        cleaning_configs=cleaning_configs,
    )


def _interactive_config_wizard() -> V10rConfig:
    """Interactive configuration wizard."""
    console.print("[blue]v10r Configuration Wizard[/blue]")
    console.print("This wizard will help you create a basic configuration.\n")
    
    # For now, return default config
    # TODO: Implement interactive wizard
    console.print("[yellow]Interactive wizard not yet implemented. Creating default config...[/yellow]")
    return _create_default_config()


def _display_config_summary(config: V10rConfig):
    """Display a summary of the configuration."""
    console.print("\n[blue]Configuration Summary[/blue]")
    
    # Databases table
    db_table = Table(title="Databases")
    db_table.add_column("Name", style="cyan")
    db_table.add_column("Host", style="magenta")
    db_table.add_column("Database", style="green")
    db_table.add_column("Tables", style="yellow")
    
    for db in config.databases:
        table_count = len(db.tables)
        db_table.add_row(
            db.name,
            db.connection.host,
            db.connection.database,
            str(table_count),
        )
    
    console.print(db_table)
    
    # Embeddings table
    emb_table = Table(title="Embedding Configurations")
    emb_table.add_column("Name", style="cyan")
    emb_table.add_column("Provider", style="magenta")
    emb_table.add_column("Model", style="green")
    emb_table.add_column("Dimensions", style="yellow")
    
    for name, emb in config.embeddings.items():
        emb_table.add_row(
            name,
            emb.provider,
            emb.model,
            str(emb.dimensions),
        )
    
    console.print(emb_table)
    
    # Tables table
    if any(db.tables for db in config.databases):
        table_table = Table(title="Tables to Vectorize")
        table_table.add_column("Database", style="cyan")
        table_table.add_column("Schema.Table", style="magenta")
        table_table.add_column("Text Column", style="green")
        table_table.add_column("Embedding Config", style="yellow")
        
        for db in config.databases:
            for table in db.tables:
                table_table.add_row(
                    db.name,
                    table.full_table_name,
                    table.text_column,
                    table.embedding_config,
                )
        
        console.print(table_table)


if __name__ == "__main__":
    main() 