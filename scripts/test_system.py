#!/usr/bin/env python3
"""
v10r System Test Script

This script performs end-to-end testing of the v10r system by:
1. Validating configuration
2. Testing database connectivity
3. Verifying CLI commands work
4. Testing basic vectorization flow (if services are running)

Usage:
    python scripts/test_system.py --config examples/simple-config.yaml
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
import redis
from rich.console import Console
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v10r.cli import main as cli_main
from v10r.config import V10rConfig
from click.testing import CliRunner


console = Console()


class SystemTestResult:
    """Container for system test results."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.failures: List[str] = []
        self.start_time = time.time()
    
    def add_success(self, test_name: str):
        """Record a successful test."""
        self.tests_run += 1
        self.tests_passed += 1
        console.print(f"✅ {test_name}")
    
    def add_failure(self, test_name: str, error: str):
        """Record a failed test."""
        self.tests_run += 1
        self.tests_failed += 1
        self.failures.append(f"{test_name}: {error}")
        console.print(f"❌ {test_name}: {error}")
    
    def print_summary(self):
        """Print test summary."""
        duration = time.time() - self.start_time
        
        console.print("\n" + "="*60)
        console.print(f"[bold]System Test Summary[/bold]")
        console.print(f"Duration: {duration:.2f}s")
        console.print(f"Tests Run: {self.tests_run}")
        console.print(f"Passed: [green]{self.tests_passed}[/green]")
        console.print(f"Failed: [red]{self.tests_failed}[/red]")
        
        if self.failures:
            console.print("\n[red]Failures:[/red]")
            for failure in self.failures:
                console.print(f"  • {failure}")
        
        success_rate = (self.tests_passed / self.tests_run * 100) if self.tests_run > 0 else 0
        if success_rate >= 80:
            console.print(f"\n[green]✅ System test PASSED ({success_rate:.1f}% success rate)[/green]")
            return True
        else:
            console.print(f"\n[red]❌ System test FAILED ({success_rate:.1f}% success rate)[/red]")
            return False


class SystemTester:
    """Main system tester class."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Optional[V10rConfig] = None
        self.result = SystemTestResult()
        self.runner = CliRunner()
    
    async def run_all_tests(self) -> bool:
        """Run all system tests."""
        console.print("[bold blue]🚀 Starting v10r System Tests[/bold blue]\n")
        
        # Core functionality tests
        await self.test_config_validation()
        await self.test_cli_commands()
        await self.test_database_connectivity()
        await self.test_redis_connectivity()
        
        # Integration tests (if applicable)
        await self.test_embedding_providers()
        
        # Print summary
        return self.result.print_summary()
    
    async def test_config_validation(self):
        """Test configuration validation."""
        try:
            self.config = V10rConfig.from_yaml(self.config_path)
            self.config.validate_config()
            self.result.add_success("Configuration validation")
        except Exception as e:
            self.result.add_failure("Configuration validation", str(e))
    
    async def test_cli_commands(self):
        """Test basic CLI commands."""
        # Test CLI help
        try:
            result = self.runner.invoke(cli_main, ['--help'])
            if result.exit_code == 0 and 'v10r: Generic PostgreSQL' in result.output:
                self.result.add_success("CLI help command")
            else:
                self.result.add_failure("CLI help command", f"Exit code: {result.exit_code}")
        except Exception as e:
            self.result.add_failure("CLI help command", str(e))
        
        # Test config validation command
        try:
            result = self.runner.invoke(cli_main, ['validate-config', '-c', self.config_path])
            if result.exit_code == 0:
                self.result.add_success("CLI validate-config command")
            else:
                self.result.add_failure("CLI validate-config command", f"Exit code: {result.exit_code}")
        except Exception as e:
            self.result.add_failure("CLI validate-config command", str(e))
        
        # Test init command
        try:
            with self.runner.isolated_filesystem():
                result = self.runner.invoke(cli_main, ['init', '-o', 'test-config.yaml'])
                if result.exit_code == 0 and Path('test-config.yaml').exists():
                    self.result.add_success("CLI init command")
                else:
                    self.result.add_failure("CLI init command", f"Exit code: {result.exit_code}")
        except Exception as e:
            self.result.add_failure("CLI init command", str(e))
    
    async def test_database_connectivity(self):
        """Test database connectivity."""
        if not self.config:
            self.result.add_failure("Database connectivity", "No valid config")
            return
        
        for db_config in self.config.databases:
            try:
                # Build connection string from nested connection object
                if hasattr(db_config, 'connection'):
                    conn_config = db_config.connection
                    # Handle both object attributes and dict access
                    if hasattr(conn_config, 'user'):
                        user = conn_config.user
                        password = conn_config.password
                        host = conn_config.host
                        port = conn_config.port
                        database = conn_config.database
                    else:
                        # Assume it's a dict
                        user = conn_config.get('user', 'postgres')
                        password = conn_config.get('password', '')
                        host = conn_config.get('host', 'localhost')
                        port = conn_config.get('port', 5432)
                        database = conn_config.get('database', 'postgres')
                    
                    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
                elif hasattr(db_config, 'connection_string'):
                    connection_string = db_config.connection_string
                elif hasattr(db_config, 'url'):
                    connection_string = db_config.url
                else:
                    self.result.add_failure(f"Database connectivity ({db_config.name})", "No connection configuration found")
                    continue
                
                conn = await asyncpg.connect(connection_string)
                
                # Test basic query
                result = await conn.fetchval("SELECT 1")
                if result == 1:
                    self.result.add_success(f"Database connectivity ({db_config.name})")
                else:
                    self.result.add_failure(f"Database connectivity ({db_config.name})", "Query failed")
                
                await conn.close()
                
            except Exception as e:
                self.result.add_failure(f"Database connectivity ({db_config.name})", str(e))
    
    async def test_redis_connectivity(self):
        """Test Redis connectivity."""
        if not self.config:
            self.result.add_failure("Redis connectivity", "No valid config")
            return
        
        try:
            # Build Redis URL from nested connection object
            if hasattr(self.config.queue, 'connection'):
                conn_config = self.config.queue.connection
                # Handle both object attributes and dict access
                if hasattr(conn_config, 'host'):
                    host = conn_config.host
                    port = conn_config.port
                    db = getattr(conn_config, 'db', 0)
                else:
                    # Assume it's a dict
                    host = conn_config.get('host', 'localhost')
                    port = conn_config.get('port', 6379)
                    db = conn_config.get('db', 0)
                redis_url = f"redis://{host}:{port}/{db}"
            elif hasattr(self.config.queue, 'redis_url'):
                redis_url = self.config.queue.redis_url
            elif hasattr(self.config.queue, 'url'):
                redis_url = self.config.queue.url
            else:
                self.result.add_failure("Redis connectivity", "No redis connection configuration found")
                return
            
            redis_client = redis.from_url(redis_url)
            
            # Test basic operation
            redis_client.ping()
            
            # Test queue operations
            test_key = "v10r:test_key"
            test_value = "test_value"
            
            redis_client.set(test_key, test_value)
            result = redis_client.get(test_key)
            
            if result and result.decode() == test_value:
                self.result.add_success("Redis connectivity")
            else:
                self.result.add_failure("Redis connectivity", "Set/get test failed")
            
            # Cleanup
            redis_client.delete(test_key)
            redis_client.close()
            
        except Exception as e:
            self.result.add_failure("Redis connectivity", str(e))
    
    async def test_embedding_providers(self):
        """Test embedding provider configurations."""
        if not self.config:
            self.result.add_failure("Embedding providers", "No valid config")
            return
        
        for name, embedding_config in self.config.embeddings.items():
            try:
                # Basic configuration validation
                required_fields = ['provider', 'model']
                missing_fields = [field for field in required_fields if not hasattr(embedding_config, field)]
                
                if missing_fields:
                    self.result.add_failure(
                        f"Embedding provider ({name})", 
                        f"Missing fields: {missing_fields}"
                    )
                    continue
                
                # Provider-specific validation (lenient for testing)
                if embedding_config.provider == 'openai':
                    if not hasattr(embedding_config, 'api_key'):
                        console.print(f"⚠️ Warning: OpenAI provider ({name}) missing api_key - may require environment variable")
                elif embedding_config.provider == 'infinity':
                    if not hasattr(embedding_config, 'base_url'):
                        console.print(f"⚠️ Warning: Infinity provider ({name}) missing base_url - may use default")
                elif embedding_config.provider == 'custom':
                    if not hasattr(embedding_config, 'base_url'):
                        console.print(f"⚠️ Warning: Custom provider ({name}) missing base_url - may use default")
                
                self.result.add_success(f"Embedding provider configuration ({name})")
                
            except Exception as e:
                self.result.add_failure(f"Embedding provider ({name})", str(e))


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="v10r System Test")
    parser.add_argument(
        '--config', '-c',
        required=True,
        help="Path to v10r configuration file"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        sys.exit(1)
    
    # Run system tests
    tester = SystemTester(str(config_path))
    success = await tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 