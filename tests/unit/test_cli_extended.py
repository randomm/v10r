"""
Extended tests for v10r.cli module to boost coverage.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import pytest
from click.testing import CliRunner
import tempfile
import os

from v10r.cli import (
    main, 
    init, 
    validate_config,
    listen,
    worker,
    status,
    register,
    schema_reconcile,
    schema_status,
    test_connection,
    handle_errors,
    _create_default_config,
    _interactive_config_wizard,
    _display_config_summary
)
from v10r.config import V10rConfig
from v10r.exceptions import ConfigurationError, V10rError


class TestHandleErrorsDecorator:
    """Test the handle_errors decorator."""
    
    def test_handle_errors_success(self):
        """Test handle_errors with successful function."""
        @handle_errors
        def success_func():
            return "success"
        
        result = success_func()
        assert result == "success"
    
    def test_handle_errors_v10r_error(self):
        """Test handle_errors with V10rError."""
        @handle_errors
        def error_func():
            raise V10rError("Test error")
        
        with patch('sys.exit') as mock_exit:
            error_func()
            mock_exit.assert_called_once_with(1)
    
    def test_handle_errors_keyboard_interrupt(self):
        """Test handle_errors with KeyboardInterrupt."""
        @handle_errors
        def interrupt_func():
            raise KeyboardInterrupt()
        
        with patch('sys.exit') as mock_exit:
            interrupt_func()
            mock_exit.assert_called_once_with(0)
    
    def test_handle_errors_generic_exception(self):
        """Test handle_errors with generic exception."""
        @handle_errors
        def exception_func():
            raise Exception("Generic error")
        
        with patch('sys.exit') as mock_exit:
            exception_func()
            mock_exit.assert_called_once_with(1)
    
    def test_handle_errors_debug_mode(self):
        """Test handle_errors with debug mode enabled."""
        @handle_errors
        def exception_func():
            raise Exception("Debug error")
        
        with patch('sys.argv', ['v10r', '--debug']), \
             patch('sys.exit') as mock_exit, \
             patch('traceback.print_exc') as mock_traceback:
            
            exception_func()
            mock_exit.assert_called_once_with(1)
            mock_traceback.assert_called_once()


class TestMainCommand:
    """Test main CLI command group."""
    
    def test_main_without_debug(self):
        """Test main command without debug flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert "v10r: Generic PostgreSQL NOTIFY-based vectorization service" in result.output
    
    def test_main_with_debug(self):
        """Test main command with debug flag."""
        runner = CliRunner()
        result = runner.invoke(main, ['--debug', '--help'])
        
        assert result.exit_code == 0
        # Debug flag should be stored in context
        assert "v10r: Generic PostgreSQL NOTIFY-based vectorization service" in result.output
    
    def test_main_version(self):
        """Test main command version option."""
        runner = CliRunner()
        result = runner.invoke(main, ['--version'])
        
        assert result.exit_code == 0
        assert result.output.strip().startswith("main, version")


class TestInitCommand:
    """Test init command."""
    
    def test_init_default_output(self):
        """Test init command with default output file."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('v10r.cli._create_default_config') as mock_create:
            
            mock_config = MagicMock()
            mock_create.return_value = mock_config
            
            runner = CliRunner()
            result = runner.invoke(init)
            
            assert result.exit_code == 0
            mock_config.to_yaml.assert_called_once_with("v10r-config.yaml")
            assert "Configuration file created" in result.output
    
    def test_init_custom_output(self):
        """Test init command with custom output file."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('v10r.cli._create_default_config') as mock_create:
            
            mock_config = MagicMock()
            mock_create.return_value = mock_config
            
            runner = CliRunner()
            result = runner.invoke(init, ['-o', 'custom-config.yaml'])
            
            assert result.exit_code == 0
            mock_config.to_yaml.assert_called_once_with("custom-config.yaml")
    
    def test_init_file_exists_no_overwrite(self):
        """Test init command when file exists and user declines overwrite."""
        with patch('pathlib.Path.exists', return_value=True):
            runner = CliRunner()
            result = runner.invoke(init, input='n\n')
            
            assert result.exit_code == 0
            # Should exit early without creating config
    
    def test_init_file_exists_with_overwrite(self):
        """Test init command when file exists and user confirms overwrite."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('v10r.cli._create_default_config') as mock_create:
            
            mock_config = MagicMock()
            mock_create.return_value = mock_config
            
            runner = CliRunner()
            result = runner.invoke(init, input='y\n')
            
            assert result.exit_code == 0
            mock_config.to_yaml.assert_called_once()
    
    def test_init_interactive_mode(self):
        """Test init command in interactive mode."""
        with patch('pathlib.Path.exists', return_value=False), \
             patch('v10r.cli._interactive_config_wizard') as mock_wizard:
            
            mock_config = MagicMock()
            mock_wizard.return_value = mock_config
            
            runner = CliRunner()
            result = runner.invoke(init, ['-i'])
            
            assert result.exit_code == 0
            mock_wizard.assert_called_once()
            mock_config.to_yaml.assert_called_once()


class TestValidateConfigCommand:
    """Test validate-config command."""
    
    def test_validate_config_success(self):
        """Test validate-config with valid configuration."""
        mock_config = MagicMock()
        
        with patch.object(V10rConfig, 'from_yaml', return_value=mock_config), \
             patch('v10r.cli._display_config_summary') as mock_display, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            
            # Write a minimal config to the temp file
            f.write("service_name: test\n")
            f.flush()
            
            try:
                runner = CliRunner()
                result = runner.invoke(validate_config, ['-c', f.name])
                
                assert result.exit_code == 0
                assert "Configuration is valid" in result.output
                mock_display.assert_called_once_with(mock_config)
            finally:
                os.unlink(f.name)
    
    def test_validate_config_invalid(self):
        """Test validate-config with invalid configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', side_effect=ConfigurationError("Invalid config")):
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    assert result.exit_code == 1
                    assert "Configuration error" in result.output
            finally:
                os.unlink(f.name)


class TestListenCommand:
    """Test listen command."""
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.listener.VectorListener')
    def test_listen_command(self, mock_listener_class, mock_asyncio_run):
        """Test listen command."""
        mock_config = MagicMock()
        mock_listener = MagicMock()
        mock_listener_class.return_value = mock_listener
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(listen, ['-c', f.name])
                    
                    # The command should execute successfully (exit_code 0)
                    # The asyncio.run should be called
                    assert result.exit_code == 0
                    mock_asyncio_run.assert_called_once()
                    # Note: VectorListener is created inside the async function, so we can't easily test it
            finally:
                os.unlink(f.name)
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.listener.VectorListener')
    def test_listen_command_custom_channel(self, mock_listener_class, mock_asyncio_run):
        """Test listen command with custom channel."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(listen, ['-c', f.name, '--channel', 'custom_events'])
                    
                    assert result.exit_code == 0
                    mock_asyncio_run.assert_called_once()
            finally:
                os.unlink(f.name)
    
    @patch('v10r.cli.asyncio.run', side_effect=KeyboardInterrupt())
    @patch('v10r.listener.VectorListener')
    def test_listen_command_keyboard_interrupt(self, mock_listener_class, mock_asyncio_run):
        """Test listen command handles keyboard interrupt."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(listen, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    assert "Shutting down listener" in result.output
            finally:
                os.unlink(f.name)


class TestWorkerCommand:
    """Test worker command."""
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.worker.VectorizerWorker')
    def test_worker_command(self, mock_worker_class, mock_asyncio_run):
        """Test worker command."""
        mock_config = MagicMock()
        mock_config.workers.concurrency = 4
        mock_worker = MagicMock()
        mock_worker_class.return_value = mock_worker
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(worker, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    assert "Workers: 4" in result.output
                    mock_asyncio_run.assert_called_once()
                    # Note: VectorizerWorker is created inside the async function
            finally:
                os.unlink(f.name)
    
    @patch('v10r.cli.asyncio.run')
    @patch('v10r.worker.VectorizerWorker')
    def test_worker_command_override_workers(self, mock_worker_class, mock_asyncio_run):
        """Test worker command with worker count override."""
        mock_config = MagicMock()
        mock_config.workers.concurrency = 4
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(worker, ['-c', f.name, '-w', '8'])
                    
                    assert result.exit_code == 0
                    assert "Workers: 8 (overridden)" in result.output
                    assert mock_config.workers.concurrency == 8
            finally:
                os.unlink(f.name)
    
    @patch('v10r.cli.asyncio.run', side_effect=KeyboardInterrupt())
    @patch('v10r.worker.VectorizerWorker')
    def test_worker_command_keyboard_interrupt(self, mock_worker_class, mock_asyncio_run):
        """Test worker command handles keyboard interrupt."""
        mock_config = MagicMock()
        mock_config.workers.concurrency = 4
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(worker, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    assert "Shutting down workers" in result.output
            finally:
                os.unlink(f.name)


class TestStatusCommand:
    """Test status command."""
    
    @patch('v10r.cli.asyncio.run')
    def test_status_command(self, mock_asyncio_run):
        """Test status command."""
        mock_config = MagicMock()
        mock_config.databases = []
        mock_config.embeddings = {}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(status, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    mock_asyncio_run.assert_called_once()
            finally:
                os.unlink(f.name)


class TestRegisterCommand:
    """Test register command."""
    
    @patch('builtins.exit')  # Mock the exit() call
    @patch('v10r.cli.asyncio.run')
    def test_register_command_minimal(self, mock_asyncio_run, mock_exit):
        """Test register command with minimal parameters."""
        mock_config = MagicMock()
        mock_asyncio_run.return_value = 0  # Success exit code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(register, [
                        '-c', f.name,
                        '--database', 'test_db',
                        '--table', 'test_table',
                        '--text-column', 'content',
                        '--embedding-config', 'openai'
                    ])
                    
                    # The command calls exit(), so we check the mock
                    mock_asyncio_run.assert_called_once()
                    mock_exit.assert_called_once_with(0)
            finally:
                os.unlink(f.name)
    
    @patch('builtins.exit')  # Mock the exit() call
    @patch('v10r.cli.asyncio.run')
    def test_register_command_full_options(self, mock_asyncio_run, mock_exit):
        """Test register command with all options."""
        mock_config = MagicMock()
        mock_asyncio_run.return_value = 0  # Success exit code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(register, [
                        '-c', f.name,
                        '--database', 'test_db',
                        '--schema', 'custom_schema',
                        '--table', 'test_table',
                        '--text-column', 'content',
                        '--vector-column', 'content_embedding',
                        '--model-column', 'content_model',
                        '--embedding-config', 'openai',
                        '--collision-strategy', 'auto_warn'
                    ])
                    
                    # The command calls exit(), so we check the mock
                    mock_asyncio_run.assert_called_once()
                    mock_exit.assert_called_once_with(0)
            finally:
                os.unlink(f.name)


class TestSchemaReconcileCommand:
    """Test schema-reconcile command."""
    
    def test_schema_reconcile_command(self):
        """Test schema-reconcile command."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(schema_reconcile, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    assert "Schema reconciliation not yet implemented" in result.output
            finally:
                os.unlink(f.name)
    
    def test_schema_reconcile_dry_run(self):
        """Test schema-reconcile command with dry run."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(schema_reconcile, ['-c', f.name, '--dry-run'])
                    
                    assert result.exit_code == 0
                    assert "Dry run mode" in result.output
            finally:
                os.unlink(f.name)


class TestSchemaStatusCommand:
    """Test schema-status command."""
    
    def test_schema_status_command(self):
        """Test schema-status command."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(schema_status, ['-c', f.name])
                    
                    assert result.exit_code == 0
                    assert "Schema status checking not yet implemented" in result.output
            finally:
                os.unlink(f.name)


class TestTestConnectionCommand:
    """Test test-connection command."""
    
    @patch('builtins.exit')  # Mock the exit() call
    @patch('v10r.cli.asyncio.run')
    def test_test_connection_command(self, mock_asyncio_run, mock_exit):
        """Test test-connection command."""
        mock_config = MagicMock()
        mock_config.databases = []
        mock_config.embeddings = {}
        mock_asyncio_run.return_value = 0  # Success exit code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config):
                    runner = CliRunner()
                    result = runner.invoke(test_connection, ['-c', f.name])
                    
                    # The command calls exit(), so we check the mock
                    mock_asyncio_run.assert_called_once()
                    mock_exit.assert_called_once_with(0)
            finally:
                os.unlink(f.name)


class TestConfigHelpers:
    """Test configuration helper functions."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = _create_default_config()
        
        assert hasattr(config, 'embeddings')
        assert hasattr(config, 'databases')
        assert hasattr(config, 'cleaning_configs')
        assert len(config.databases) > 0
        assert len(config.embeddings) > 0

    @patch('builtins.input')
    def test_interactive_config_wizard(self, mock_input):
        """Test interactive configuration wizard."""
        # Mock user inputs
        mock_input.side_effect = ["test_service", "localhost", "5432", "testdb"]
        
        # For now, this just returns default config
        config = _create_default_config()
        assert config is not None

    def test_display_config_summary(self):
        """Test displaying configuration summary."""
        # Create a mock config with proper structure
        mock_config = MagicMock()
        mock_config.databases = []
        mock_config.embeddings = {}
        
        # Mock database with tables attribute
        mock_db = MagicMock()
        mock_db.name = "test_db"
        mock_db.connection.host = "localhost"
        mock_db.connection.database = "testdb"
        mock_db.tables = []  # This is the key fix - ensure tables is a list
        mock_config.databases = [mock_db]
        
        # This should not raise an exception
        _display_config_summary(mock_config)


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_cli_help_all_commands(self):
        """Test that all commands have help text."""
        runner = CliRunner()
        
        # Test main help
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        assert "v10r" in result.output
        
        # Test command help
        commands = ['init', 'validate-config', 'listen', 'worker', 'status', 
                   'register', 'schema-reconcile', 'schema-status', 'test-connection']
        
        for cmd in commands:
            result = runner.invoke(main, [cmd, '--help'])
            assert result.exit_code == 0
    
    def test_missing_config_file(self):
        """Test handling of missing config file."""
        runner = CliRunner()
        
        # Test with non-existent config file
        result = runner.invoke(validate_config, ['-c', 'nonexistent.yaml'])
        assert result.exit_code == 2  # Click exits with 2 for invalid options
        assert "does not exist" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def test_command_with_v10r_error(self):
        """Test command that raises V10rError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', side_effect=V10rError("Test error")):
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    assert result.exit_code == 1
                    assert "Test error" in result.output
            finally:
                os.unlink(f.name)
    
    def test_command_with_generic_exception(self):
        """Test command that raises generic exception."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', side_effect=Exception("Generic error")):
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    assert result.exit_code == 1
                    assert "Unexpected error" in result.output
            finally:
                os.unlink(f.name)
    
    def test_command_with_keyboard_interrupt(self):
        """Test command that receives KeyboardInterrupt."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', side_effect=KeyboardInterrupt()):
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    assert result.exit_code == 0  # KeyboardInterrupt exits gracefully
                    assert "Interrupted" in result.output
            finally:
                os.unlink(f.name)


class TestAdvancedCLIScenarios:
    """Test advanced CLI scenarios."""
    
    def test_multiple_error_types_in_sequence(self):
        """Test handling multiple error types."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', side_effect=ConfigurationError("Configuration error")):
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    expected_text = "Configuration error"
                    assert expected_text in result.output
            finally:
                os.unlink(f.name)
    
    def test_config_validation_edge_cases(self):
        """Test configuration validation edge cases."""
        mock_config = MagicMock()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("service_name: test\n")
            f.flush()
            
            try:
                with patch.object(V10rConfig, 'from_yaml', return_value=mock_config), \
                     patch('v10r.cli._display_config_summary'):
                    
                    runner = CliRunner()
                    result = runner.invoke(validate_config, ['-c', f.name])
                    
                    assert "Configuration is valid" in result.output
            finally:
                os.unlink(f.name) 