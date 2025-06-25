"""
Unit tests for the v10r CLI interface.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from click.testing import CliRunner
from v10r.cli import main, handle_errors
from v10r.config import V10rConfig
from v10r.exceptions import ConfigurationError


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
databases:
  - name: test_db
    connection:
      host: localhost
      port: 5432
      database: test
      user: user
      password: pass
    tables:
      - schema: public
        table: test_table
        text_column: content
        embedding_config: test_embedding

queue:
  type: redis
  connection:
    host: localhost
    port: 6379
    db: 0
  queue_name: v10r:embedding_queue

embeddings:
  test_embedding:
    provider: openai
    model: text-embedding-3-small
    api_key: test-key
    dimensions: 1536

workers:
  concurrency: 4
  batch_timeout: 60
""")
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestCLIMain:
    """Test main CLI functionality."""
    
    def test_cli_help(self, runner):
        """Test CLI help output."""
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert 'v10r: Generic PostgreSQL NOTIFY-based vectorization service' in result.output
    
    def test_cli_version(self, runner):
        """Test CLI version output."""
        result = runner.invoke(main, ['--version'])
        assert result.exit_code == 0
        # Should show version info
    
    def test_cli_debug_mode(self, runner):
        """Test CLI debug mode flag."""
        result = runner.invoke(main, ['--debug', '--help'])
        assert result.exit_code == 0
        # Debug mode should not affect help output


class TestInitCommand:
    """Test init command functionality."""
    
    def test_init_default_output(self, runner):
        """Test init command with default output file."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init'])
            assert result.exit_code == 0
            assert 'Configuration file created: v10r-config.yaml' in result.output
    
    def test_init_custom_output(self, runner):
        """Test init command with custom output file."""
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init', '-o', 'custom-config.yaml'])
            assert result.exit_code == 0
            assert 'Configuration file created: custom-config.yaml' in result.output
    
    @patch('v10r.cli.click.confirm')
    def test_init_file_exists_no_overwrite(self, mock_confirm, runner):
        """Test init command when file exists and user says no to overwrite."""
        mock_confirm.return_value = False
        
        with runner.isolated_filesystem():
            # Create existing file
            with open('test-config.yaml', 'w') as f:
                f.write('existing content')
            
            result = runner.invoke(main, ['init', '-o', 'test-config.yaml'])
            assert result.exit_code == 0
            # File should still have original content
            with open('test-config.yaml', 'r') as f:
                assert f.read() == 'existing content'
    
    @patch('v10r.cli.click.confirm')
    def test_init_file_exists_overwrite(self, mock_confirm, runner):
        """Test init command when file exists and user says yes to overwrite."""
        mock_confirm.return_value = True
        
        with runner.isolated_filesystem():
            # Create existing file
            with open('test-config.yaml', 'w') as f:
                f.write('existing content')
            
            result = runner.invoke(main, ['init', '-o', 'test-config.yaml'])
            assert result.exit_code == 0
            # File should be overwritten
            with open('test-config.yaml', 'r') as f:
                content = f.read()
                assert 'existing content' not in content
                assert 'databases' in content  # Should have new config
    
    @patch('v10r.cli._interactive_config_wizard')
    def test_init_interactive_mode(self, mock_wizard, runner):
        """Test init command with interactive mode."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_wizard.return_value = mock_config
        
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init', '-i', '-o', 'interactive-config.yaml'])
            assert result.exit_code == 0
            mock_wizard.assert_called_once()


class TestValidateConfigCommand:
    """Test validate-config command functionality."""
    
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_validate_config_valid_file(self, mock_from_yaml, runner, temp_config_file):
        """Test validate-config with valid configuration file."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.validate_config.return_value = None
        
        # Set up mock config attributes for _display_config_summary
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_config.workers = MagicMock()
        mock_config.workers.concurrency = 4
        
        mock_from_yaml.return_value = mock_config
        
        result = runner.invoke(main, ['validate-config', '-c', temp_config_file])
        assert result.exit_code == 0
        assert 'Configuration is valid' in result.output
        mock_config.validate_config.assert_called_once()
    
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_validate_config_invalid_file(self, mock_from_yaml, runner, temp_config_file):
        """Test validate-config with invalid configuration file."""
        mock_from_yaml.side_effect = ConfigurationError("Test error")
        
        result = runner.invoke(main, ['validate-config', '-c', temp_config_file])
        assert result.exit_code == 1
        assert 'Configuration error: Test error' in result.output
    
    def test_validate_config_file_not_exists(self, runner):
        """Test validate-config with non-existent file."""
        result = runner.invoke(main, ['validate-config', '-c', 'nonexistent.yaml'])
        assert result.exit_code != 0


class TestListenCommand:
    """Test listen command functionality."""
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_listen_command(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test listen command execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return None (simulating successful execution)
        mock_asyncio_run.return_value = None
        
        result = runner.invoke(main, ['listen', '-c', temp_config_file])
        assert result.exit_code == 0
        
        # Verify that config was loaded and asyncio.run was called
        mock_from_yaml.assert_called_once_with(temp_config_file)
        mock_asyncio_run.assert_called_once()
        
        # Verify the command output contains expected information (strip ANSI codes)
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Starting v10r listener service' in output_plain
        assert 'Channel: v10r_events' in output_plain
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_listen_command_custom_channel(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test listen command with custom channel."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return None (simulating successful execution)
        mock_asyncio_run.return_value = None
        
        result = runner.invoke(main, ['listen', '-c', temp_config_file, '--channel', 'custom_channel'])
        assert result.exit_code == 0
        
        # Verify that config was loaded and asyncio.run was called
        mock_from_yaml.assert_called_once_with(temp_config_file)
        mock_asyncio_run.assert_called_once()
        
        # Verify the command output contains expected information (strip ANSI codes)
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Starting v10r listener service' in output_plain
        assert 'Channel: custom_channel' in output_plain

    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_listen_command_keyboard_interrupt(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test listen command with keyboard interrupt."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to raise KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(main, ['listen', '-c', temp_config_file])
        assert result.exit_code == 0  # Should handle gracefully
        
        # Verify the command output contains shutdown message
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Shutting down listener...' in output_plain


class TestWorkerCommand:
    """Test worker command functionality."""
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_worker_command(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test worker command execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.workers = MagicMock()
        mock_config.workers.concurrency = 4
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return None (simulating successful execution)
        mock_asyncio_run.return_value = None
        
        result = runner.invoke(main, ['worker', '-c', temp_config_file])
        assert result.exit_code == 0
        
        # Verify that config was loaded and asyncio.run was called
        mock_from_yaml.assert_called_once_with(temp_config_file)
        mock_asyncio_run.assert_called_once()
        
        # Verify the command output contains expected information
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())  
        assert 'Starting v10r worker service' in output_plain
        assert 'Workers: 4' in output_plain
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_worker_command_override_workers(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test worker command with worker override."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.workers = MagicMock()
        mock_config.workers.concurrency = 4
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return None (simulating successful execution)
        mock_asyncio_run.return_value = None
        
        result = runner.invoke(main, ['worker', '-c', temp_config_file, '-w', '8'])
        assert result.exit_code == 0
        
        # Verify the config was modified
        assert mock_config.workers.concurrency == 8
        
        # Verify the command output
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Workers: 8 (overridden)' in output_plain

    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_worker_command_keyboard_interrupt(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test worker command with keyboard interrupt."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.workers = MagicMock()
        mock_config.workers.concurrency = 4
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to raise KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(main, ['worker', '-c', temp_config_file])
        assert result.exit_code == 0  # Should handle gracefully
        
        # Verify the command output contains shutdown message
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Shutting down workers...' in output_plain


class TestStatusCommand:
    """Test status command functionality."""
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_status_command_success(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test status command successful execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return None (simulating successful execution)
        mock_asyncio_run.return_value = None
        
        result = runner.invoke(main, ['status', '-c', temp_config_file])
        assert result.exit_code == 0
        
        # Verify the command output contains expected information
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'v10r Service Status' in output_plain

    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_status_command_keyboard_interrupt(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test status command with keyboard interrupt."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to raise KeyboardInterrupt
        mock_asyncio_run.side_effect = KeyboardInterrupt()
        
        result = runner.invoke(main, ['status', '-c', temp_config_file])
        assert result.exit_code == 1  # Should exit with error code
        
        # Verify the command output contains cancellation message
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Status check cancelled' in output_plain

    @patch('v10r.cli.asyncio.run')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_status_command_error(self, mock_from_yaml, mock_asyncio_run, runner, temp_config_file):
        """Test status command with error."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to raise generic exception
        mock_asyncio_run.side_effect = Exception("Test error")
        
        result = runner.invoke(main, ['status', '-c', temp_config_file])
        assert result.exit_code == 1  # Should exit with error code
        
        # Verify the command output contains error message
        output_plain = ''.join(c for c in result.output if ord(c) < 128 and c.isprintable() or c.isspace())
        assert 'Status check error: Test error' in output_plain


class TestRegisterCommand:
    """Test register command functionality."""
    
    @patch('v10r.registration.TableRegistrationService')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_register_command_success(self, mock_from_yaml, mock_service_class, runner, temp_config_file):
        """Test register command successful execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        # Mock the result of the async register_table call
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.vector_column = "content_vector"
        mock_result.model_column = "content_model"
        mock_result.timestamp_column = None
        mock_result.warnings = []
        mock_result.message = "Registration successful"
        
        async def mock_register_table(*args, **kwargs):
            return mock_result
        
        mock_service.register_table = mock_register_table
        
        # Mock asyncio.run to actually run the coroutine
        with patch('v10r.cli.asyncio.run') as mock_asyncio_run:
            def run_coro(coro):
                # Create a new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_asyncio_run.side_effect = run_coro
            
            result = runner.invoke(main, [
                'register', 
                '-c', temp_config_file, 
                '--database', 'test_db',
                '--table', 'test_table',
                '--text-column', 'content',
                '--embedding-config', 'test_embedding'
            ])
            
            # Check that the command exited with code 0 (success)
            assert result.exit_code == 0
            assert 'registered successfully' in result.output
    
    @patch('v10r.registration.TableRegistrationService')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_register_command_with_optional_params(self, mock_from_yaml, mock_service_class, runner, temp_config_file):
        """Test register command with optional parameters."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        # Mock the result of the async register_table call
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.vector_column = "custom_vector"
        mock_result.model_column = "custom_model"
        mock_result.timestamp_column = None
        mock_result.warnings = []
        mock_result.message = "Registration successful"
        
        async def mock_register_table(*args, **kwargs):
            return mock_result
        
        mock_service.register_table = mock_register_table
        
        # Mock asyncio.run to actually run the coroutine
        with patch('v10r.cli.asyncio.run') as mock_asyncio_run:
            def run_coro(coro):
                # Create a new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_asyncio_run.side_effect = run_coro
            
            result = runner.invoke(main, [
                'register',
                '-c', temp_config_file,
                '--database', 'test_db',
                '--table', 'test_table',
                '--text-column', 'content',
                '--vector-column', 'custom_vector',
                '--embedding-config', 'test_embedding'
            ])
            
            # Check that the command exited with code 0 (success)
            assert result.exit_code == 0
            assert 'registered successfully' in result.output

    @patch('v10r.registration.TableRegistrationService')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_register_command_failure(self, mock_from_yaml, mock_service_class, runner, temp_config_file):
        """Test register command when registration fails."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        # Mock the result of the async register_table call - failure case
        mock_result = MagicMock()
        mock_result.success = False
        mock_result.error = "Table not found"
        
        async def mock_register_table(*args, **kwargs):
            return mock_result
        
        mock_service.register_table = mock_register_table
        
        # Mock asyncio.run to actually run the coroutine
        with patch('v10r.cli.asyncio.run') as mock_asyncio_run:
            def run_coro(coro):
                # Create a new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_asyncio_run.side_effect = run_coro
            
            result = runner.invoke(main, [
                'register',
                '-c', temp_config_file,
                '--database', 'test_db',
                '--table', 'test_table',
                '--text-column', 'content',
                '--embedding-config', 'test_embedding'
            ])
            
            # Check that the command exited with code 1 (failure)
            assert result.exit_code == 1
            assert 'Registration failed' in result.output

    @patch('v10r.registration.TableRegistrationService')
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_register_command_with_warnings(self, mock_from_yaml, mock_service_class, runner, temp_config_file):
        """Test register command with warnings."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        
        # Mock the result with warnings
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.vector_column = "content_vector"
        mock_result.model_column = "content_model"
        mock_result.timestamp_column = "created_at"
        mock_result.warnings = ["Column already exists", "Using existing index"]
        
        async def mock_register_table(*args, **kwargs):
            return mock_result
        
        mock_service.register_table = mock_register_table
        
        # Mock asyncio.run to actually run the coroutine
        with patch('v10r.cli.asyncio.run') as mock_asyncio_run:
            def run_coro(coro):
                # Create a new event loop and run the coroutine
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            mock_asyncio_run.side_effect = run_coro
            
            result = runner.invoke(main, [
                'register',
                '-c', temp_config_file,
                '--database', 'test_db',
                '--table', 'test_table',
                '--text-column', 'content',
                '--embedding-config', 'test_embedding'
            ])
            
            # Check success with warnings
            assert result.exit_code == 0
            assert 'registered successfully' in result.output
            assert 'Warnings:' in result.output
            assert 'Column already exists' in result.output


class TestSchemaReconcileCommand:
    """Test schema-reconcile command functionality."""
    
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_schema_reconcile_command(self, mock_from_yaml, runner, temp_config_file):
        """Test schema-reconcile command execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        result = runner.invoke(main, ['schema-reconcile', '-c', temp_config_file])
        assert result.exit_code == 0
        
        # Verify that config was loaded
        mock_from_yaml.assert_called_once_with(temp_config_file)
        
        # Verify the output contains the expected message
        assert 'Schema reconciliation not yet implemented' in result.output

    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_schema_reconcile_command_dry_run(self, mock_from_yaml, runner, temp_config_file):
        """Test schema-reconcile command with dry run."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        result = runner.invoke(main, ['schema-reconcile', '-c', temp_config_file, '--dry-run'])
        assert result.exit_code == 0
        
        # Verify that config was loaded
        mock_from_yaml.assert_called_once_with(temp_config_file)
        
        # Verify the output contains the expected messages
        assert 'Dry run mode - no changes will be made' in result.output
        assert 'Schema reconciliation not yet implemented' in result.output


class TestSchemaStatusCommand:
    """Test schema-status command functionality."""
    
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_schema_status_command(self, mock_from_yaml, runner, temp_config_file):
        """Test schema-status command execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_from_yaml.return_value = mock_config
        
        result = runner.invoke(main, ['schema-status', '-c', temp_config_file])
        assert result.exit_code == 0
        
        # Verify that config was loaded
        mock_from_yaml.assert_called_once_with(temp_config_file)
        
        # Verify the output contains the expected message
        assert 'Schema status checking not yet implemented' in result.output


class TestTestConnectionCommand:
    """Test test-connection command functionality."""
    
    @patch('v10r.cli.V10rConfig.from_yaml')
    def test_test_connection_command_basic(self, mock_from_yaml, runner, temp_config_file):
        """Test test-connection command basic execution."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.databases = []  # Empty to avoid actual connections
        mock_config.embeddings = {}  # Empty to avoid actual API calls
        mock_from_yaml.return_value = mock_config
        
        # Mock asyncio.run to return success
        with patch('v10r.cli.asyncio.run') as mock_asyncio_run:
            mock_asyncio_run.return_value = 0  # Success
            
            result = runner.invoke(main, ['test-connection', '-c', temp_config_file])
            
            # Should exit with code 0 since we mocked success
            assert result.exit_code == 0
            assert 'Testing connections' in result.output


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_handle_errors_decorator_v10r_error(self, runner, temp_config_file):
        """Test handle_errors decorator with v10r exception."""
        result = runner.invoke(main, ['validate-config', '-c', temp_config_file])
        # Should not fail with unhandled exception
        assert result.exit_code in [0, 1]  # Either success or handled error
    
    def test_handle_errors_decorator_keyboard_interrupt(self, runner, temp_config_file):
        """Test handle_errors decorator with KeyboardInterrupt."""
        with patch('v10r.cli.V10rConfig.from_yaml', side_effect=KeyboardInterrupt()):
            result = runner.invoke(main, ['validate-config', '-c', temp_config_file])
            # The handle_errors decorator catches KeyboardInterrupt and handles it gracefully
            assert result.exit_code == 0
            assert 'Operation cancelled' in result.output or 'Interrupted' in result.output
    
    def test_handle_errors_decorator_unexpected_error(self, runner, temp_config_file):
        """Test handle_errors decorator with unexpected error."""
        with patch('v10r.cli.V10rConfig.from_yaml', side_effect=Exception("Unexpected error")):
            result = runner.invoke(main, ['validate-config', '-c', temp_config_file])
            assert result.exit_code == 1
            assert 'Unexpected error: Unexpected error' in result.output
    
    def test_handle_errors_decorator_debug_mode(self, runner, temp_config_file):
        """Test handle_errors decorator in debug mode."""
        with patch('v10r.cli.V10rConfig.from_yaml', side_effect=Exception("Debug error")):
            result = runner.invoke(main, ['--debug', 'validate-config', '-c', temp_config_file])
            assert result.exit_code == 1
            # In debug mode, should show the error message
            assert 'Debug error' in result.output


class TestErrorHandlerFunction:
    """Test the error handler function directly."""
    
    def test_handle_errors_decorator_wraps_function(self):
        """Test that handle_errors decorator properly wraps functions."""
        @handle_errors
        def test_function():
            return "success"
        
        assert test_function() == "success"
    
    def test_handle_errors_decorator_handles_exceptions(self):
        """Test that handle_errors decorator handles exceptions."""
        
        @handle_errors  
        def failing_function():
            raise ValueError("Test error")
        
        # Should not raise exception, but handle it gracefully
        # The decorator should handle the exception and exit
        with pytest.raises(SystemExit):
            failing_function()


class TestConfigHelpers:
    """Test configuration helper functions."""
    
    @patch('v10r.cli._create_default_config')
    def test_create_default_config_called(self, mock_create_default):
        """Test that _create_default_config is called for non-interactive init."""
        mock_config = MagicMock(spec=V10rConfig)
        mock_create_default.return_value = mock_config
        
        runner = CliRunner()
        with runner.isolated_filesystem():
            result = runner.invoke(main, ['init', '-o', 'test-config.yaml'])
            mock_create_default.assert_called_once()
    
    @patch('v10r.cli._display_config_summary')
    def test_display_config_summary_called(self, mock_display):
        """Test that _display_config_summary is called during validation."""
        runner = CliRunner()
        
        with patch('v10r.cli.V10rConfig.from_yaml') as mock_from_yaml:
            mock_config = MagicMock(spec=V10rConfig)
            mock_config.validate_config.return_value = None
            mock_from_yaml.return_value = mock_config
            
            with runner.isolated_filesystem():
                with open('test-config.yaml', 'w') as f:
                    f.write('test: config')
                
                result = runner.invoke(main, ['validate-config', '-c', 'test-config.yaml'])
                mock_display.assert_called_once_with(mock_config)

    def test_create_default_config_function(self):
        """Test the _create_default_config function directly."""
        from v10r.cli import _create_default_config
        
        config = _create_default_config()
        assert isinstance(config, V10rConfig)
        assert len(config.databases) == 1
        assert config.databases[0].name == "primary"
        assert config.databases[0].connection.host == "${POSTGRES_HOST}"

    def test_display_config_summary_function(self):
        """Test the _display_config_summary function directly."""
        from v10r.cli import _display_config_summary
        
        # Create a mock config
        mock_config = MagicMock(spec=V10rConfig)
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_config.workers = MagicMock()
        mock_config.workers.concurrency = 4
        
        # This should not raise an exception
        _display_config_summary(mock_config)

    @patch('v10r.cli.click.prompt')
    @patch('v10r.cli.click.confirm')
    def test_interactive_config_wizard_basic(self, mock_confirm, mock_prompt):
        """Test basic interactive config wizard functionality."""
        from v10r.cli import _interactive_config_wizard
        
        # Mock user inputs
        mock_prompt.side_effect = [
            'test_db',           # database name
            'localhost',         # host
            '5432',             # port
            'testdb',           # database
            'testuser',         # user
            'testpass',         # password
            'openai',           # provider
            'sk-test-key',      # api key
            'text-embedding-3-small',  # model
            '1536'              # dimensions
        ]
        mock_confirm.side_effect = [False]  # Don't add more databases
        
        config = _interactive_config_wizard()
        assert isinstance(config, V10rConfig)

    def test_display_config_summary_with_data(self):
        """Test _display_config_summary with actual config data."""
        from v10r.cli import _display_config_summary
        from v10r.config import DatabaseConfig, DatabaseConnection, EmbeddingConfig, WorkerConfig
        
        # Create a real config with data
        db_config = DatabaseConfig(
            name="test_db",
            connection=DatabaseConnection(
                host="localhost",
                port=5432,
                database="test",
                user="user",
                password="pass"
            )
        )
        
        embedding_config = EmbeddingConfig(
            provider="openai",
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536
        )
        
        worker_config = WorkerConfig(concurrency=4)
        
        config = V10rConfig(
            databases=[db_config],
            embeddings={"default": embedding_config},
            workers=worker_config
        )
        
        # This should not raise an exception and should display info
        _display_config_summary(config) 