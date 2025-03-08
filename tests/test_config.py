import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from qurious.config import Config, ModelConfig, PathConfig, TrainingConfig


class TestConfig(unittest.TestCase):
    def test_default_config(self):
        """Test that default configuration values are set correctly."""
        config = Config()

        # Test model config defaults
        self.assertEqual(config.model.base_model, "meta-llama/Llama-3.2-3B-Instruct")
        self.assertTrue(config.model.lora_enabled)
        self.assertEqual(config.model.lora_config.r, 8)

        # Test training config defaults
        self.assertEqual(config.training.epochs, 1)
        self.assertEqual(config.training.batch_size, 4)
        self.assertEqual(config.training.learning_rate, 1e-4)

        # Test path config defaults
        self.assertEqual(config.paths.output_dir, "./outputs")
        self.assertEqual(config.paths.log_dir, "./logs")
        self.assertEqual(config.paths.data_dir, "./data")

    def test_float_with_scientific_notation(self):
        """Test that float values with scientific notation are parsed correctly."""
        config = Config()
        config.training.learning_rate = 1e-5
        self.assertEqual(config.training.learning_rate, 1e-5)

        # Test that strings are converted to floats
        config = Config(training={"learning_rate": "1e-5"})
        self.assertEqual(config.training.learning_rate, 1e-5)

    def test_override_partial_config_in_constructor(self):
        config = Config(training={"learning_rate": 2e-4})
        self.assertEqual(config.training.learning_rate, 2e-4)
        self.assertEqual(config.training.batch_size, 4)  # Default value

    def test_load_from_yaml(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            config_data = {
                "model": {"base_model": "gpt2", "lora_enabled": False, "lora_config": {"r": 16}},
                "training": {"epochs": 3, "batch_size": 8},
                "paths": {"output_dir": "./custom_outputs"},
            }
            with open(tmp_file.name, "w") as yaml_file:
                # Write the YAML data to the temporary file
                yaml.dump(config_data, yaml_file)
            tmp_file_path = tmp_file.name

        try:
            config = Config.from_yaml(tmp_file_path)

            # Check that values were loaded correctly
            self.assertEqual(config.model.base_model, "gpt2")
            self.assertFalse(config.model.lora_enabled)
            self.assertEqual(config.model.lora_config.r, 16)
            self.assertEqual(config.training.epochs, 3)
            self.assertEqual(config.training.batch_size, 8)
            self.assertEqual(config.paths.output_dir, "./custom_outputs")

            # Check that unspecified values use defaults
            self.assertEqual(config.training.learning_rate, 1e-4)
            self.assertEqual(config.paths.log_dir, "./logs")
        finally:
            os.unlink(tmp_file_path)

    def test_save_to_yaml(self):
        """Test saving configuration to YAML file."""
        config = Config()
        config.model.base_model = "custom-model"
        config.training.epochs = 5

        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "test_config.yaml"
            config.save_yaml(yaml_path)

            # Check that file was created
            self.assertTrue(yaml_path.exists())

            # Load the saved file and verify contents
            with open(yaml_path, "r") as f:
                loaded_config = yaml.safe_load(f)

            self.assertEqual(loaded_config["model"]["base_model"], "custom-model")
            self.assertEqual(loaded_config["training"]["epochs"], 5)

    def test_file_not_found(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Config.from_yaml("/path/to/nonexistent/config.yaml")

    def test_env_var_override(self):
        """Test that environment variables override config values."""
        with patch.dict(
            os.environ,
            {
                "QURIOUS_MODEL__BASE_MODEL": "env-model",
                "QURIOUS_TRAINING__BATCH_SIZE": "16",
                "QURIOUS_PATHS__OUTPUT_DIR": "/custom/path",
            },
        ):
            config = Config()
            self.assertEqual(config.model.base_model, "env-model")
            self.assertEqual(config.training.batch_size, 16)
            self.assertEqual(config.paths.output_dir, "/custom/path")

    def test_nested_env_var_override(self):
        """Test that nested environment variables override nested config values."""
        with patch.dict(os.environ, {"QURIOUS_MODEL__LORA_ENABLED": "false", "QURIOUS_MODEL__LORA_CONFIG__R": "32"}):
            config = Config()
            self.assertFalse(config.model.lora_enabled)
            self.assertEqual(config.model.lora_config.r, 32)

    def test_individual_configs(self):
        """Test creating individual config objects."""
        model_config = ModelConfig(base_model="custom-model")
        self.assertEqual(model_config.base_model, "custom-model")

        training_config = TrainingConfig(epochs=10, learning_rate=2e-5)
        self.assertEqual(training_config.epochs, 10)
        self.assertEqual(training_config.learning_rate, 2e-5)

        path_config = PathConfig(output_dir="./custom_path")
        self.assertEqual(path_config.output_dir, "./custom_path")


if __name__ == "__main__":
    unittest.main()
