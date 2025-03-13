from unittest.mock import MagicMock, patch

import pytest
import torch

from qurious.config import Config
from qurious.llms.lora_manager import LoraManager


# Create a fixture for a basic config
@pytest.fixture
def mock_config():
    # Create a Config object with the expected structure
    config = Config(
        {
            "model": {
                "base_model": "mock-model/test",
                "device": "cpu",
                "lora_enabled": True,
                "lora_config": {
                    "r": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "target_modules": "all-linear",
                    "bias": "none",
                    "task_type": "CAUSAL_LM",
                },
            },
            "training": {"checkpoint_dir": "./test_checkpoints"},
        }
    )
    return config


# Patch the main dependencies
@pytest.fixture
def mock_dependencies():
    with (
        patch("qurious.llms.lora_manager.AutoModelForCausalLM") as mock_model_cls,
        patch("qurious.llms.lora_manager.AutoTokenizer") as mock_tokenizer_cls,
    ):
        # Setup the mock model
        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        # Setup the mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Setup mock peft model
        mock_peft_model = MagicMock()
        mock_peft_model.active_adapter = "default"

        # Setup mock peft config
        mock_config_inst = MagicMock()

        yield {
            "model_cls": mock_model_cls,
            "model": mock_model,
            "tokenizer_cls": mock_tokenizer_cls,
            "tokenizer": mock_tokenizer,
            "peft_model": mock_peft_model,
            "peft_config_inst": mock_config_inst,
        }


class TestLoraManager:
    """Tests for the LoraManager class."""

    def test_init(self, mock_config, mock_dependencies):
        """Test initialization of LoraManager."""
        manager = LoraManager(mock_config)

        # Check base model loading
        mock_dependencies["model_cls"].from_pretrained.assert_called_once()

        # Check tokenizer loading
        mock_dependencies["tokenizer_cls"].from_pretrained.assert_called_once()

        # Check attributes
        assert manager.base_name == mock_config.model.base_model
        assert manager.device == torch.device("cpu")

    def test_add_adapter_with_config(self, mock_config, mock_dependencies):
        """Test adding an adapter with an explicit config."""
        with patch("qurious.llms.lora_manager.PeftLoraConfig"):
            # Setup
            manager = LoraManager(mock_config)
            mock_config_obj = MagicMock()

            # Execute
            manager.add_adapter("test-adapter", mock_config_obj)

            # Assert
            manager.model.add_adapter.assert_called_with(mock_config_obj, "test-adapter")

    def test_add_adapter_with_default_config(self, mock_config, mock_dependencies):
        """Test adding an adapter with the default config."""
        with patch.object(LoraManager, "_create_peft_config") as mock_create_config:
            # Setup
            manager = LoraManager(mock_config)
            mock_peft_config = MagicMock()
            mock_create_config.return_value = mock_peft_config

            # Execute
            manager.add_adapter("test-adapter")

            # Assert
            mock_create_config.assert_called()
            manager.model.add_adapter.assert_called_with(mock_peft_config, "test-adapter")

    def test_create_default_adapter_when_enabled(self, mock_config, mock_dependencies):
        """Test that default adapter is created when LoRA is enabled."""
        with patch.object(LoraManager, "_create_default_adapter") as mock_create_default:
            # Ensure lora_enabled is True
            mock_config.model.lora_enabled = True

            # Initialize manager
            LoraManager(mock_config)

            # Check that _create_default_adapter was called
            mock_create_default.assert_called_once()

    def test_no_default_adapter_when_disabled(self, mock_config, mock_dependencies):
        """Test that default adapter is not created when LoRA is disabled."""
        with patch.object(LoraManager, "_create_default_adapter") as mock_create_default:
            # Set lora_enabled to False
            mock_config.model.lora_enabled = False

            # Initialize manager
            LoraManager(mock_config)

            # Check that _create_default_adapter was not called
            mock_create_default.assert_not_called()

    def test_get_base_model(self, mock_config, mock_dependencies):
        """Test getting the base model without adapters."""
        # Setup
        manager = LoraManager(mock_config)

        # Execute
        result = manager.get_base_model()

        # Assert
        manager.model.disable_adapters.assert_called_once()
        assert result == manager.model

    def test_get_model_with_adapter(self, mock_config, mock_dependencies):
        """Test getting the model with a specific adapter."""
        # Setup
        manager = LoraManager(mock_config)
        adapter_name = "test-adapter"

        # Execute
        result = manager.get_model(adapter_name)

        # Assert
        manager.model.set_adapter.assert_called_once_with(adapter_name)
        assert result == manager.model
