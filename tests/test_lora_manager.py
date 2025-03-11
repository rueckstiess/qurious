from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig as PeftLoraConfig

from qurious.config import Config
from qurious.llms.lora_manager import LoraManager


# Create a fixture for a basic config
@pytest.fixture
def mock_config():
    # Create a Config object with the expected structure
    config = Config({
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
                "task_type": "CAUSAL_LM"
            }
        },
        "paths": {
            "checkpoint_dir": "./test_checkpoints"
        }
    })
    return config


# Patch the main dependencies
@pytest.fixture
def mock_dependencies():
    with (
        patch("qurious.llms.lora_manager.AutoModelForCausalLM") as mock_model_cls,
        patch("qurious.llms.lora_manager.AutoTokenizer") as mock_tokenizer_cls,
        patch("qurious.llms.lora_manager.get_peft_model") as mock_get_peft,
        patch("qurious.llms.lora_manager.load_peft_weights") as mock_load_weights,
        patch("qurious.llms.lora_manager.PeftConfig") as mock_peft_config,
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
        mock_get_peft.return_value = mock_peft_model

        # Setup mock peft config
        mock_config_inst = MagicMock()
        mock_peft_config.from_pretrained.return_value = mock_config_inst

        yield {
            "model_cls": mock_model_cls,
            "model": mock_model,
            "tokenizer_cls": mock_tokenizer_cls,
            "tokenizer": mock_tokenizer,
            "get_peft": mock_get_peft,
            "peft_model": mock_peft_model,
            "load_weights": mock_load_weights,
            "peft_config": mock_peft_config,
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

        # Check default adapter creation
        mock_dependencies["get_peft"].assert_called_once()

        # Check attributes
        assert manager.base_name == mock_config.model.base_model
        assert manager.device == torch.device("cpu")
        assert "default" in manager.adapters

    def test_init_without_lora(self, mock_config, mock_dependencies):
        """Test initialization with LoRA disabled."""
        mock_config.model.lora_enabled = False
        manager = LoraManager(mock_config)

        # Check that no adapters were created
        mock_dependencies["get_peft"].assert_not_called()
        assert len(manager.adapters) == 0

    def test_add_adapter(self, mock_config, mock_dependencies):
        """Test adding a new adapter."""
        manager = LoraManager(mock_config)

        # Reset the mock to clear the default adapter creation call
        mock_dependencies["get_peft"].reset_mock()

        # Add a new adapter
        manager.add_adapter("test_adapter")

        # Check that get_peft_model was called again
        mock_dependencies["get_peft"].assert_called_once()

        # Check that the adapter was added
        assert "test_adapter" in manager.adapters

    def test_add_adapter_with_custom_config(self, mock_config, mock_dependencies):
        """Test adding a new adapter with a custom config."""
        manager = LoraManager(mock_config)

        # Reset the mock to clear the default adapter creation call
        mock_dependencies["get_peft"].reset_mock()

        # Create a custom config
        custom_config = PeftLoraConfig(r=16, lora_alpha=32, target_modules="all-linear")

        # Add a new adapter with the custom config
        manager.add_adapter("custom_adapter", custom_config)

        # Check that get_peft_model was called with the custom config
        mock_dependencies["get_peft"].assert_called_once_with(manager.base_model, custom_config)

        # Check that the adapter was added
        assert "custom_adapter" in manager.adapters

    def test_add_adapter_already_exists(self, mock_config, mock_dependencies):
        """Test adding an adapter that already exists."""
        manager = LoraManager(mock_config)

        # Try to add an adapter with the same name as the default
        with pytest.raises(ValueError, match=r"Adapter 'default' already exists"):
            manager.add_adapter("default")

    def test_remove_adapter(self, mock_config, mock_dependencies):
        """Test removing an adapter."""
        manager = LoraManager(mock_config)

        # Check that the default adapter exists
        assert "default" in manager.adapters

        # Remove the default adapter
        manager.remove_adapter("default")

        # Check that the adapter was removed
        assert "default" not in manager.adapters

    def test_remove_nonexistent_adapter(self, mock_config, mock_dependencies):
        """Test removing an adapter that doesn't exist."""
        manager = LoraManager(mock_config)

        # Try to remove a non-existent adapter
        with pytest.raises(ValueError, match=r"Adapter 'nonexistent' does not exist"):
            manager.remove_adapter("nonexistent")

    def test_copy_adapter(self, mock_config, mock_dependencies):
        """Test copying an adapter."""
        manager = LoraManager(mock_config)

        # Setup the mock peft model to have a config
        mock_dependencies["peft_model"].peft_config = {"default": MagicMock()}

        # Reset the get_peft mock to clear the default adapter creation call
        mock_dependencies["get_peft"].reset_mock()

        # Copy the default adapter
        manager.copy_adapter("default", "copy_adapter")

        # Check that get_peft_model was called again
        mock_dependencies["get_peft"].assert_called_once()

        # Check that the adapter was copied
        assert "copy_adapter" in manager.adapters

    def test_copy_nonexistent_adapter(self, mock_config, mock_dependencies):
        """Test copying an adapter that doesn't exist."""
        manager = LoraManager(mock_config)

        # Try to copy a non-existent adapter
        with pytest.raises(ValueError, match=r"Source adapter 'nonexistent' does not exist"):
            manager.copy_adapter("nonexistent", "copy_adapter")

    def test_copy_to_existing_adapter(self, mock_config, mock_dependencies):
        """Test copying to an adapter name that already exists."""
        manager = LoraManager(mock_config)

        # Setup another adapter
        manager.adapters["another"] = MagicMock()

        # Try to copy to an existing adapter name
        with pytest.raises(ValueError, match=r"Target adapter 'another' already exists"):
            manager.copy_adapter("default", "another")

    def test_get_base_model(self, mock_config, mock_dependencies):
        """Test getting the base model."""
        manager = LoraManager(mock_config)

        # Get the base model
        model = manager.get_base_model()

        # Check that the base model was returned
        assert model == manager.base_model

    def test_get_model_adapter(self, mock_config, mock_dependencies):
        """Test getting a model with an adapter."""
        manager = LoraManager(mock_config)

        # Get the model with the default adapter
        model = manager.get_model("default")

        # Check that the adapter model was returned
        assert model == manager.adapters["default"]

    def test_get_model_nonexistent_adapter(self, mock_config, mock_dependencies):
        """Test getting a model with an adapter that doesn't exist."""
        manager = LoraManager(mock_config)

        # Try to get a model with a non-existent adapter
        with pytest.raises(ValueError, match=r"Adapter 'nonexistent' does not exist"):
            manager.get_model("nonexistent")

    def test_list_adapters(self, mock_config, mock_dependencies):
        """Test listing adapters."""
        manager = LoraManager(mock_config)

        # Add another adapter
        manager.adapters["another"] = MagicMock()

        # List the adapters
        adapters = manager.list_adapters()

        # Check that both adapters are listed
        assert set(adapters) == {"default", "another"}

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_save_adapter(self, mock_exists, mock_mkdir, mock_config, mock_dependencies):
        """Test saving an adapter."""
        mock_exists.return_value = True
        manager = LoraManager(mock_config)

        # Save the default adapter
        manager.save_adapter("default")

        # Check that the directory was created
        mock_mkdir.assert_called_once()

        # Check that the adapter was saved
        mock_dependencies["peft_model"].save_pretrained.assert_called_once()

    def test_save_nonexistent_adapter(self, mock_config, mock_dependencies):
        """Test saving an adapter that doesn't exist."""
        manager = LoraManager(mock_config)

        # Try to save a non-existent adapter
        with pytest.raises(ValueError, match=r"Adapter 'nonexistent' does not exist"):
            manager.save_adapter("nonexistent")

    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    def test_save_all_adapters(self, mock_exists, mock_mkdir, mock_config, mock_dependencies):
        """Test saving all adapters."""
        mock_exists.return_value = True
        manager = LoraManager(mock_config)

        # Add another adapter - use the same mock to track calls correctly
        another_adapter = MagicMock()
        manager.adapters["another"] = another_adapter

        # Save all adapters
        with patch.object(manager, "save_adapter") as mock_save_adapter:
            manager.save_all_adapters()

            # Check that save_adapter was called for each adapter
            assert mock_save_adapter.call_count == 2
            mock_save_adapter.assert_any_call("default", Path(mock_config.paths.checkpoint_dir) / "default")
            mock_save_adapter.assert_any_call("another", Path(mock_config.paths.checkpoint_dir) / "another")

    @patch("pathlib.Path.exists")
    def test_load_adapter(self, mock_exists, mock_config, mock_dependencies):
        """Test loading an adapter."""
        mock_exists.return_value = True
        manager = LoraManager(mock_config)

        # Remove the default adapter to test loading into an empty slot
        del manager.adapters["default"]

        # Reset the mocks
        mock_dependencies["get_peft"].reset_mock()
        mock_dependencies["load_weights"].reset_mock()

        # Load an adapter
        manager.load_adapter("loaded_adapter", "./test_path")

        # Check that PeftConfig.from_pretrained was called
        mock_dependencies["peft_config"].from_pretrained.assert_called_once_with(Path("./test_path"))

        # Check that get_peft_model was called
        mock_dependencies["get_peft"].assert_called_once()

        # Check that load_peft_weights was called
        mock_dependencies["load_weights"].assert_called_once()

        # Check that the adapter was added
        assert "loaded_adapter" in manager.adapters

    def test_load_adapter_already_exists(self, mock_config, mock_dependencies):
        """Test loading an adapter with a name that already exists."""
        manager = LoraManager(mock_config)

        # Try to load an adapter with the same name as the default
        with pytest.raises(ValueError, match=r"Adapter 'default' already exists"):
            manager.load_adapter("default", "./test_path")

    @patch("pathlib.Path.exists")
    def test_load_adapter_path_not_found(self, mock_exists, mock_config, mock_dependencies):
        """Test loading an adapter from a non-existent path."""
        mock_exists.return_value = False
        manager = LoraManager(mock_config)

        # Try to load from a non-existent path
        with pytest.raises(FileNotFoundError):
            manager.load_adapter("new_adapter", "./nonexistent_path")

    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.iterdir")
    def test_load_all_adapters(self, mock_iterdir, mock_exists, mock_config, mock_dependencies):
        """Test loading all adapters."""
        mock_exists.return_value = True

        # Setup mock directories
        adapter_dir1 = MagicMock(spec=Path)
        adapter_dir1.name = "adapter1"
        adapter_dir1.is_dir.return_value = True
        adapter_config_file = MagicMock(spec=Path)
        adapter_config_file.name = "adapter_config.json"
        adapter_config_file.exists.return_value = True
        # Need to use __truediv__ since / is overloaded for Path objects
        adapter_dir1.__truediv__.return_value = adapter_config_file

        adapter_dir2 = MagicMock(spec=Path)
        adapter_dir2.name = "adapter2"
        adapter_dir2.is_dir.return_value = True
        adapter_dir2.__truediv__.return_value = adapter_config_file

        mock_iterdir.return_value = [adapter_dir1, adapter_dir2]

        manager = LoraManager(mock_config)

        # Remove default adapter to avoid conflicts
        del manager.adapters["default"]

        # Reset mocks
        mock_dependencies["get_peft"].reset_mock()
        mock_dependencies["load_weights"].reset_mock()

        # Create a spy on load_adapter
        with patch.object(manager, "load_adapter") as mock_load_adapter:
            # Load all adapters
            manager.load_all_adapters("./test_base_path")

            # Check that load_adapter was called for each adapter
            assert mock_load_adapter.call_count == 2
            mock_load_adapter.assert_any_call("adapter1", adapter_dir1)
            mock_load_adapter.assert_any_call("adapter2", adapter_dir2)

    @patch("pathlib.Path.exists")
    def test_load_all_adapters_path_not_found(self, mock_exists, mock_config, mock_dependencies):
        """Test loading all adapters from a non-existent path."""
        mock_exists.return_value = False
        manager = LoraManager(mock_config)

        # Try to load from a non-existent path
        with pytest.raises(FileNotFoundError):
            manager.load_all_adapters("./nonexistent_path")
