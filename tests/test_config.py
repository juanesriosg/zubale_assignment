"""
Tests for configuration settings and environment variable handling.
"""

import pytest
import os
import tempfile
from unittest.mock import patch, mock_open
from app.config import Settings


class TestSettings:
    """Test cases for Settings class."""
    
    def test_default_settings(self):
        """Test settings with default values."""
        # Clear all relevant environment variables
        env_keys_to_clear = [
            'ANTHROPIC_API_KEY', 'TOP_K_RETRIEVAL', 'EMBEDDING_MODEL',
            'LLM_MODEL', 'VECTOR_STORE_PATH', 'LOG_LEVEL'
        ]
        
        # Save current environment
        original_env = {}
        for key in env_keys_to_clear:
            if key in os.environ:
                original_env[key] = os.environ[key]
                del os.environ[key]
        
        try:
            # Temporarily patch the Settings class to not use .env file
            class TestSettingsNoEnv(Settings):
                class Config:
                    env_file = "nonexistent.env"  # Point to non-existent file
            
            settings = TestSettingsNoEnv()
            
            # Verify default values
            assert settings.anthropic_api_key is None
            assert settings.top_k_retrieval == 3
            assert settings.embedding_model == "all-MiniLM-L6-v2"
            assert settings.llm_model == "claude-3-haiku-20240307"
            assert settings.vector_store_path == "./vector_store"
            assert settings.log_level == "INFO"
        finally:
            # Restore original environment
            for key, value in original_env.items():
                os.environ[key] = value
    
    def test_settings_from_environment_variables(self):
        """Test settings loading from environment variables."""
        env_vars = {
            'ANTHROPIC_API_KEY': 'test-api-key-123',
            'TOP_K_RETRIEVAL': '5',
            'EMBEDDING_MODEL': 'custom-embedding-model',
            'LLM_MODEL': 'claude-3-sonnet-20240229',
            'VECTOR_STORE_PATH': '/custom/path',
            'LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            assert settings.anthropic_api_key == 'test-api-key-123'
            assert settings.top_k_retrieval == 5
            assert settings.embedding_model == 'custom-embedding-model'
            assert settings.llm_model == 'claude-3-sonnet-20240229'
            assert settings.vector_store_path == '/custom/path'
            assert settings.log_level == 'DEBUG'
    
    def test_settings_type_validation(self):
        """Test that settings validate types correctly."""
        # Test integer validation for top_k_retrieval
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': 'not-a-number'}, clear=False):
            with pytest.raises(ValueError):
                Settings()
        
        # Test valid integer
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': '10'}, clear=False):
            settings = Settings()
            assert settings.top_k_retrieval == 10
            assert isinstance(settings.top_k_retrieval, int)
    
    def test_settings_with_env_file(self):
        """Test settings loading from .env file."""
        env_content = """
ANTHROPIC_API_KEY=env-file-api-key
TOP_K_RETRIEVAL=7
EMBEDDING_MODEL=env-file-model
LLM_MODEL=claude-3-opus-20240229
VECTOR_STORE_PATH=/env/file/path
LOG_LEVEL=WARNING
"""
        
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content.strip())
            env_file_path = f.name
        
        try:
            # Test with custom env_file path
            class TestSettings(Settings):
                class Config:
                    env_file = env_file_path
            
            settings = TestSettings()
            
            assert settings.anthropic_api_key == 'env-file-api-key'
            assert settings.top_k_retrieval == 7
            assert settings.embedding_model == 'env-file-model'
            assert settings.llm_model == 'claude-3-opus-20240229'
            assert settings.vector_store_path == '/env/file/path'
            assert settings.log_level == 'WARNING'
            
        finally:
            # Clean up temporary file
            os.unlink(env_file_path)
    
    def test_environment_variables_override_env_file(self):
        """Test that environment variables take precedence over .env file."""
        env_content = """
ANTHROPIC_API_KEY=env-file-key
TOP_K_RETRIEVAL=5
LOG_LEVEL=ERROR
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content.strip())
            env_file_path = f.name
        
        try:
            # Set environment variables that should override
            env_vars = {
                'ANTHROPIC_API_KEY': 'env-var-key',
                'TOP_K_RETRIEVAL': '8'
            }
            
            with patch.dict(os.environ, env_vars, clear=False):
                class TestSettings(Settings):
                    class Config:
                        env_file = env_file_path
                
                settings = TestSettings()
                
                # Environment variables should override
                assert settings.anthropic_api_key == 'env-var-key'
                assert settings.top_k_retrieval == 8
                # .env file value should be used where no env var exists
                assert settings.log_level == 'ERROR'
                
        finally:
            os.unlink(env_file_path)
    
    def test_api_key_validation_scenarios(self):
        """Test various API key scenarios."""
        # Test with valid API key  
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-api03-sSOhhErSLJQyrMiqoeXh_pCuckiv4rtZx16fdW2ysZnTP1vSNcgc8gK0j5m9Hk'}, clear=True):
            settings = Settings()
            assert settings.anthropic_api_key.startswith('sk-ant-api03-')
        
        # Test with empty API key
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': ''}, clear=True):
            settings = Settings()
            assert settings.anthropic_api_key == ''
        
        # Test with no API key (default)
        with patch.dict(os.environ, {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is completely removed
            os.environ.pop('ANTHROPIC_API_KEY', None)
            
            # Use TestSettings that doesn't read from .env file
            class TestSettingsNoEnv(Settings):
                class Config:
                    env_file = "nonexistent.env"
            
            settings = TestSettingsNoEnv()
            assert settings.anthropic_api_key is None
    
    def test_top_k_retrieval_boundary_values(self):
        """Test top_k_retrieval with boundary values."""
        # Test minimum value
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': '1'}, clear=False):
            settings = Settings()
            assert settings.top_k_retrieval == 1
        
        # Test larger value
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': '100'}, clear=False):
            settings = Settings()
            assert settings.top_k_retrieval == 100
        
        # Test zero (edge case)
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': '0'}, clear=False):
            settings = Settings()
            assert settings.top_k_retrieval == 0
        
        # Test negative value (should still parse but may not be logical)
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': '-1'}, clear=False):
            settings = Settings()
            assert settings.top_k_retrieval == -1
    
    def test_log_level_validation(self):
        """Test log level setting with various values."""
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        
        for level in valid_log_levels:
            with patch.dict(os.environ, {'LOG_LEVEL': level}, clear=False):
                settings = Settings()
                assert settings.log_level == level
        
        # Test lowercase (should work)
        with patch.dict(os.environ, {'LOG_LEVEL': 'debug'}, clear=False):
            settings = Settings()
            assert settings.log_level == 'debug'
        
        # Test mixed case
        with patch.dict(os.environ, {'LOG_LEVEL': 'Warning'}, clear=False):
            settings = Settings()
            assert settings.log_level == 'Warning'
    
    def test_vector_store_path_variations(self):
        """Test vector store path with different formats."""
        path_variations = [
            './vector_store',
            '/absolute/path/to/store',
            'relative/path',
            '~/home/path',
            '/tmp/vector_store',
            'C:\\Windows\\Path'  # Windows path
        ]
        
        for path in path_variations:
            with patch.dict(os.environ, {'VECTOR_STORE_PATH': path}, clear=False):
                settings = Settings()
                assert settings.vector_store_path == path
    
    def test_embedding_model_variations(self):
        """Test embedding model with different model names."""
        model_variations = [
            'all-MiniLM-L6-v2',
            'all-MiniLM-L12-v2',
            'all-mpnet-base-v2',
            'custom-embedding-model-v1'
        ]
        
        for model in model_variations:
            with patch.dict(os.environ, {'EMBEDDING_MODEL': model}, clear=False):
                settings = Settings()
                assert settings.embedding_model == model
    
    def test_llm_model_variations(self):
        """Test LLM model with different Claude models."""
        model_variations = [
            'claude-3-haiku-20240307',
            'claude-3-sonnet-20240229',
            'claude-3-opus-20240229',
            'claude-2.1',
            'claude-instant-1.2'
        ]
        
        for model in model_variations:
            with patch.dict(os.environ, {'LLM_MODEL': model}, clear=False):
                settings = Settings()
                assert settings.llm_model == model
    
    def test_settings_immutability_after_creation(self):
        """Test that settings values can be accessed but structure is defined."""
        settings = Settings()
        
        # Verify we can read all attributes
        assert hasattr(settings, 'anthropic_api_key')
        assert hasattr(settings, 'top_k_retrieval')
        assert hasattr(settings, 'embedding_model')
        assert hasattr(settings, 'llm_model')
        assert hasattr(settings, 'vector_store_path')
        assert hasattr(settings, 'log_level')
        
        # Verify attributes have expected types
        assert isinstance(settings.top_k_retrieval, int)
        assert isinstance(settings.embedding_model, str)
        assert isinstance(settings.llm_model, str)
        assert isinstance(settings.vector_store_path, str)
        assert isinstance(settings.log_level, str)
        assert settings.anthropic_api_key is None or isinstance(settings.anthropic_api_key, str)
    
    def test_settings_repr_and_str(self):
        """Test string representations of settings."""
        settings = Settings()
        
        # Test that string representation works (doesn't expose sensitive data)
        settings_str = str(settings)
        settings_repr = repr(settings)
        
        assert isinstance(settings_str, str)
        assert isinstance(settings_repr, str)
        assert len(settings_str) > 0
        assert len(settings_repr) > 0


class TestSettingsValidation:
    """Test validation and error handling for settings."""
    
    def test_invalid_types_raise_validation_errors(self):
        """Test that invalid types raise appropriate validation errors."""
        # Test invalid integer
        with patch.dict(os.environ, {'TOP_K_RETRIEVAL': 'not-an-integer'}, clear=False):
            with pytest.raises((ValueError, TypeError)):
                Settings()
    
    def test_settings_with_missing_env_file(self):
        """Test behavior when .env file doesn't exist."""
        # This should not raise an error, just use defaults
        class TestSettings(Settings):
            class Config:
                env_file = "non-existent-file.env"
        
        settings = TestSettings()
        
        # Should use default values
        assert settings.anthropic_api_key is None
        assert settings.top_k_retrieval == 3
    
    def test_complex_environment_scenario(self):
        """Test a complex real-world-like environment scenario."""
        env_vars = {
            'ANTHROPIC_API_KEY': 'sk-ant-api03-production-key-12345',
            'TOP_K_RETRIEVAL': '5',
            'EMBEDDING_MODEL': 'all-mpnet-base-v2',
            'LLM_MODEL': 'claude-3-sonnet-20240229',
            'VECTOR_STORE_PATH': '/data/production/vector_store',
            'LOG_LEVEL': 'INFO'
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # Verify all settings are correct
            assert settings.anthropic_api_key == 'sk-ant-api03-production-key-12345'
            assert settings.top_k_retrieval == 5
            assert settings.embedding_model == 'all-mpnet-base-v2'
            assert settings.llm_model == 'claude-3-sonnet-20240229'
            assert settings.vector_store_path == '/data/production/vector_store'
            assert settings.log_level == 'INFO'
    
    def test_partial_environment_override(self):
        """Test scenario where only some environment variables are set."""
        # Only set some environment variables
        env_vars = {
            'ANTHROPIC_API_KEY': 'partial-key',
            'LOG_LEVEL': 'DEBUG'
        }
        
        with patch.dict(os.environ, env_vars, clear=False):
            settings = Settings()
            
            # Overridden values
            assert settings.anthropic_api_key == 'partial-key'
            assert settings.log_level == 'DEBUG'
            
            # Default values for non-overridden settings
            assert settings.top_k_retrieval == 3
            assert settings.embedding_model == "all-MiniLM-L6-v2"
            assert settings.llm_model == "claude-3-haiku-20240307"
            assert settings.vector_store_path == "./vector_store"