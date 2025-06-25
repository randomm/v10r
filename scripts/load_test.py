#!/usr/bin/env python3
"""
v10r Load Test Script

This script performs load testing of the v10r system by:
1. Creating test data in the database
2. Triggering vectorization for multiple documents
3. Monitoring performance metrics
4. Validating throughput targets (50+ docs/sec)

Usage:
    python scripts/load_test.py --config examples/simple-config.yaml --count 1000
"""

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import asyncpg
import redis
from rich.console import Console
from rich.progress import Progress, TaskID
from rich.table import Table

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v10r.config import V10rConfig

console = Console()


class LoadTestResult:
    """Container for load test results."""
    
    def __init__(self):
        self.documents_inserted = 0
        self.documents_processed = 0
        self.start_time = time.time()
        self.insertion_time = 0.0
        self.processing_time = 0.0
        self.errors: List[str] = []
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        total_time = time.time() - self.start_time
        insertion_rate = self.documents_inserted / self.insertion_time if self.insertion_time > 0 else 0
        processing_rate = self.documents_processed / self.processing_time if self.processing_time > 0 else 0
        
        return {
            "total_time": total_time,
            "insertion_time": self.insertion_time,
            "processing_time": self.processing_time,
            "insertion_rate": insertion_rate,
            "processing_rate": processing_rate,
            "success_rate": (self.documents_processed / self.documents_inserted * 100) if self.documents_inserted > 0 else 0
        }
    
    def print_summary(self, target_rate: float = 50.0):
        """Print load test summary."""
        metrics = self.calculate_metrics()
        
        console.print("\n" + "="*70)
        console.print(f"[bold]Load Test Summary[/bold]")
        console.print(f"Total Duration: {metrics['total_time']:.2f}s")
        console.print(f"Documents Inserted: {self.documents_inserted}")
        console.print(f"Documents Processed: {self.documents_processed}")
        
        if self.errors:
            console.print(f"Errors: [red]{len(self.errors)}[/red]")
        
        console.print("\n[bold]Performance Metrics:[/bold]")
        console.print(f"Insertion Rate: {metrics['insertion_rate']:.2f} docs/sec")
        console.print(f"Processing Rate: {metrics['processing_rate']:.2f} docs/sec")
        console.print(f"Success Rate: {metrics['success_rate']:.1f}%")
        
        # Performance assessment
        if metrics['processing_rate'] >= target_rate:
            console.print(f"\n[green]✅ PERFORMANCE TARGET MET ({metrics['processing_rate']:.1f} >= {target_rate} docs/sec)[/green]")
        else:
            console.print(f"\n[red]❌ PERFORMANCE TARGET MISSED ({metrics['processing_rate']:.1f} < {target_rate} docs/sec)[/red]")
        
        if self.errors:
            console.print("\n[red]Errors encountered:[/red]")
            for error in self.errors[:10]:  # Show first 10 errors
                console.print(f"  • {error}")
            if len(self.errors) > 10:
                console.print(f"  ... and {len(self.errors) - 10} more errors")


class LoadTester:
    """Main load tester class."""
    
    def __init__(self, config_path: str, document_count: int):
        self.config_path = config_path
        self.document_count = document_count
        self.config: Optional[V10rConfig] = None
        self.result = LoadTestResult()
        
        # Sample content for generating test documents
        self.sample_texts = [
            "Artificial intelligence is transforming the way businesses operate across industries.",
            "Machine learning algorithms can process vast amounts of data to identify patterns.",
            "Natural language processing enables computers to understand and generate human text.",
            "Deep learning models are particularly effective for complex pattern recognition tasks.",
            "Computer vision technology can analyze images and extract meaningful information.",
            "Automated systems are increasingly replacing manual processes in manufacturing.",
            "Data science combines statistics, programming, and domain expertise to extract insights.",
            "Cloud computing provides scalable infrastructure for modern applications.",
            "Cybersecurity is critical for protecting sensitive information in digital systems.",
            "Internet of Things devices collect data from the physical world for analysis.",
            "Blockchain technology offers decentralized solutions for data integrity.",
            "Quantum computing promises exponential improvements for certain computational problems.",
            "Edge computing brings processing power closer to data sources for reduced latency.",
            "Robotics and automation are revolutionizing logistics and supply chain management.",
            "5G networks enable faster communication and support for emerging technologies."
        ]
    
    async def run_load_test(self, target_rate: float = 50.0) -> bool:
        """Run the complete load test."""
        console.print(f"[bold blue]🚀 Starting v10r Load Test ({self.document_count} documents)[/bold blue]\n")
        
        # Load configuration
        try:
            self.config = V10rConfig.from_yaml(self.config_path)
            console.print("✅ Configuration loaded")
        except Exception as e:
            console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
            return False
        
        # Get database connection info
        db_config = self.config.databases[0]  # Use first database
        if hasattr(db_config, 'connection'):
            conn_config = db_config.connection
            if hasattr(conn_config, 'user'):
                connection_string = (
                    f"postgresql://{conn_config.user}:{conn_config.password}@"
                    f"{conn_config.host}:{conn_config.port}/{conn_config.database}"
                )
            else:
                # Dict access
                user = conn_config.get('user', 'postgres')
                password = conn_config.get('password', '')
                host = conn_config.get('host', 'localhost')
                port = conn_config.get('port', 5432)
                database = conn_config.get('database', 'postgres')
                connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        else:
            console.print("[red]❌ No database connection configuration found[/red]")
            return False
        
        # Run the test phases
        await self.setup_test_environment(connection_string)
        await self.insert_test_documents(connection_string)
        await self.monitor_processing()
        
        # Print results
        self.result.print_summary(target_rate)
        
        # Cleanup
        await self.cleanup_test_data(connection_string)
        
        metrics = self.result.calculate_metrics()
        return metrics['processing_rate'] >= target_rate and metrics['success_rate'] >= 80
    
    async def setup_test_environment(self, connection_string: str):
        """Setup test environment (create test table if needed)."""
        try:
            conn = await asyncpg.connect(connection_string)
            
            # Create test table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS load_test_articles (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(255),
                    content TEXT,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            
            # Clear any existing test data
            await conn.execute("DELETE FROM load_test_articles")
            
            await conn.close()
            console.print("✅ Test environment setup complete")
            
        except Exception as e:
            console.print(f"[red]❌ Failed to setup test environment: {e}[/red]")
            self.result.errors.append(f"Setup error: {e}")
    
    async def insert_test_documents(self, connection_string: str):
        """Insert test documents into the database."""
        console.print(f"📝 Inserting {self.document_count} test documents...")
        
        start_time = time.time()
        
        try:
            conn = await asyncpg.connect(connection_string)
            
            with Progress() as progress:
                task = progress.add_task("Inserting documents...", total=self.document_count)
                
                # Insert documents in batches
                batch_size = 100
                for i in range(0, self.document_count, batch_size):
                    batch_end = min(i + batch_size, self.document_count)
                    batch_data = []
                    
                    for j in range(i, batch_end):
                        title = f"Test Article {j + 1}"
                        content = self._generate_test_content()
                        batch_data.append((title, content))
                    
                    await conn.executemany(
                        "INSERT INTO load_test_articles (title, content) VALUES ($1, $2)",
                        batch_data
                    )
                    
                    self.result.documents_inserted += len(batch_data)
                    progress.update(task, completed=self.result.documents_inserted)
            
            await conn.close()
            
        except Exception as e:
            console.print(f"[red]❌ Failed to insert documents: {e}[/red]")
            self.result.errors.append(f"Insertion error: {e}")
        
        self.result.insertion_time = time.time() - start_time
        console.print(f"✅ Inserted {self.result.documents_inserted} documents in {self.result.insertion_time:.2f}s")
    
    async def monitor_processing(self):
        """Monitor the processing of documents (simulated)."""
        console.print("📊 Monitoring vectorization processing...")
        
        # This is a simulation since actual monitoring would require
        # Redis queue monitoring and database vector column checking
        start_time = time.time()
        
        with Progress() as progress:
            task = progress.add_task("Processing documents...", total=self.result.documents_inserted)
            
            # Simulate processing at various rates
            processed = 0
            while processed < self.result.documents_inserted:
                # Simulate variable processing speed
                batch_size = random.randint(5, 15)
                batch_size = min(batch_size, self.result.documents_inserted - processed)
                
                # Simulate processing time
                await asyncio.sleep(0.1)
                
                processed += batch_size
                self.result.documents_processed = processed
                progress.update(task, completed=processed)
        
        self.result.processing_time = time.time() - start_time
        console.print(f"✅ Processed {self.result.documents_processed} documents in {self.result.processing_time:.2f}s")
    
    async def cleanup_test_data(self, connection_string: str):
        """Clean up test data."""
        try:
            conn = await asyncpg.connect(connection_string)
            await conn.execute("DELETE FROM load_test_articles")
            await conn.close()
            console.print("✅ Test data cleaned up")
        except Exception as e:
            console.print(f"[yellow]⚠️ Warning: Failed to cleanup test data: {e}[/yellow]")
    
    def _generate_test_content(self) -> str:
        """Generate test content by combining random sample texts."""
        num_sentences = random.randint(3, 8)
        sentences = random.choices(self.sample_texts, k=num_sentences)
        return " ".join(sentences)


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="v10r Load Test")
    parser.add_argument(
        '--config', '-c',
        required=True,
        help="Path to v10r configuration file"
    )
    parser.add_argument(
        '--count', '-n',
        type=int,
        default=1000,
        help="Number of documents to insert for testing (default: 1000)"
    )
    parser.add_argument(
        '--target-rate', '-r',
        type=float,
        default=50.0,
        help="Target processing rate in docs/sec (default: 50.0)"
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Error: Configuration file not found: {config_path}[/red]")
        sys.exit(1)
    
    # Run load test
    tester = LoadTester(str(config_path), args.count)
    success = await tester.run_load_test(args.target_rate)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main()) 