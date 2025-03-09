from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from peft import (
    LoraConfig as PeftLoraConfig,
)
from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model,
    load_peft_weights,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from qurious.config import Config
from qurious.utils import auto_device


class LoraManager:
    """
    Manager class for handling LoRA adapters with language models.

    This class allows loading, saving, and hot-swapping between different LoRA adapters
    while maintaining a single instance of the base model in memory.
    """

    def __init__(self, config: Config):
        """
        Initialize the LoRA manager with a base model and default adapter.

        Args:
            config: Configuration object containing model and LoRA settings
        """
        self.config = config
        self.base_name = config.model.base_model
        self.lora_config = config.model.lora_config
        self.device = self._resolve_device(config.model.device)

        # Load the base model and tokenizer
        self.base_model = self._load_base_model()
        self.tokenizer = self._load_tokenizer()

        # Initialize adapters dictionary
        self.adapters: Dict[str, PeftModel] = {}

        # Create default adapter if LoRA is enabled
        if self.config.model.lora_enabled:
            self._create_default_adapter()

    def _resolve_device(self, device_str: str) -> torch.device:
        """Resolve the device string to a torch.device."""
        if device_str == "auto":
            return auto_device()
        return torch.device(device_str)

    def _load_base_model(self) -> PreTrainedModel:
        """Load the base pretrained model from Hugging Face."""
        print(f"Loading base model: {self.base_name}")
        model = AutoModelForCausalLM.from_pretrained(
            self.base_name,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map=self.device if self.device.type == "cuda" else None,
        )

        if self.device.type == "cpu":
            model = model.to(self.device)

        return model

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load the tokenizer associated with the model."""
        tokenizer = AutoTokenizer.from_pretrained(self.base_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer

    def _create_peft_config(self) -> PeftLoraConfig:
        """Convert our LoraConfig to PEFT's LoraConfig."""
        return PeftLoraConfig(**self.lora_config.model_dump())

    def _create_default_adapter(self) -> None:
        """Create the default LoRA adapter."""
        peft_config = self._create_peft_config()
        self.add_adapter("default", peft_config)

    def add_adapter(self, name: str, config: Optional[PeftLoraConfig] = None) -> None:
        """
        Add a new LoRA adapter.

        Args:
            name: Name of the adapter
            config: Optional LoRA configuration. If None, uses the default config.

        Raises:
            ValueError: If an adapter with the given name already exists
        """
        if name in self.adapters:
            raise ValueError(f"Adapter '{name}' already exists")

        if config is None:
            config = self._create_peft_config()

        print(f"Creating adapter: {name}")
        adapter_model = get_peft_model(self.base_model, config)
        self.adapters[name] = adapter_model

    def remove_adapter(self, name: str) -> None:
        """
        Remove a LoRA adapter.

        Args:
            name: Name of the adapter to remove

        Raises:
            ValueError: If the adapter doesn't exist
        """
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' does not exist")

        # Remove the adapter from memory
        del self.adapters[name]

    def copy_adapter(self, source_name: str, target_name: str) -> None:
        """
        Copy an existing adapter to a new name.

        Args:
            source_name: Name of the source adapter
            target_name: Name for the new adapter

        Raises:
            ValueError: If the source adapter doesn't exist or the target name is already used
        """
        if source_name not in self.adapters:
            raise ValueError(f"Source adapter '{source_name}' does not exist")

        if target_name in self.adapters:
            raise ValueError(f"Target adapter '{target_name}' already exists")

        # Create a new adapter with the same configuration
        source_adapter = self.adapters[source_name]
        source_config = source_adapter.peft_config[source_adapter.active_adapter]

        # Add the new adapter
        self.add_adapter(target_name, source_config)

        # Copy weights from source to target
        for source_param, target_param in zip(
            self.adapters[source_name].parameters(), self.adapters[target_name].parameters()
        ):
            with torch.no_grad():
                target_param.data.copy_(source_param.data)

    def get_model(self, adapter: Optional[str] = None) -> Union[PreTrainedModel, PeftModel]:
        """
        Get the model with the specified adapter or the base model if None.

        Args:
            adapter: Name of the adapter to use, or None for the base model

        Returns:
            The model with the specified adapter applied, or the base model

        Raises:
            ValueError: If the specified adapter doesn't exist
        """
        if adapter is None:
            return self.base_model

        if adapter not in self.adapters:
            raise ValueError(f"Adapter '{adapter}' does not exist")

        return self.adapters[adapter]

    def list_adapters(self) -> List[str]:
        """Get a list of all available adapter names."""
        return list(self.adapters.keys())

    def save_adapter(self, adapter_name: str, save_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save a specific adapter to disk.

        Args:
            adapter_name: Name of the adapter to save
            save_path: Path to save the adapter. If None, uses the default path from config.

        Raises:
            ValueError: If the adapter doesn't exist
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' does not exist")

        # Determine save path
        if save_path is None:
            save_path = Path(self.config.paths.checkpoint_dir) / adapter_name
        else:
            save_path = Path(save_path)

        # Ensure the directory exists
        save_path.mkdir(parents=True, exist_ok=True)

        # Save the adapter
        adapter_model = self.adapters[adapter_name]
        adapter_model.save_pretrained(save_path)
        print(f"Saved adapter '{adapter_name}' to {save_path}")

    def save_all_adapters(self, base_path: Optional[Union[str, Path]] = None) -> None:
        """
        Save all adapters to disk.

        Args:
            base_path: Base directory to save adapters. If None, uses the default from config.
        """
        if base_path is None:
            base_path = Path(self.config.paths.checkpoint_dir)
        else:
            base_path = Path(base_path)

        for adapter_name in self.adapters:
            adapter_path = base_path / adapter_name
            self.save_adapter(adapter_name, adapter_path)

    def load_adapter(
        self,
        adapter_name: str,
        load_path: Optional[Union[str, Path]] = None,
        adapter_name_in_folder: Optional[str] = None,
    ) -> None:
        """
        Load an adapter from disk.

        Args:
            adapter_name: Name to give the loaded adapter
            load_path: Path to load the adapter from. If None, uses the default path from config.
            adapter_name_in_folder: Name of the adapter in the saved folder if different from adapter_name

        Raises:
            ValueError: If an adapter with the given name already exists
        """
        if adapter_name in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' already exists")

        # Determine load path
        if load_path is None:
            load_path = Path(self.config.paths.checkpoint_dir) / adapter_name
        else:
            load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Adapter path '{load_path}' does not exist")

        # Load the PEFT config from disk
        peft_config = PeftConfig.from_pretrained(load_path)

        # Create a new adapter with the loaded config
        adapter_model = get_peft_model(self.base_model, peft_config)

        # Load the adapter weights
        adapter_name_to_load = adapter_name_in_folder or adapter_model.active_adapter
        load_peft_weights(adapter_model, load_path, adapter_name=adapter_name_to_load)

        # Store the adapter
        self.adapters[adapter_name] = adapter_model
        print(f"Loaded adapter '{adapter_name}' from {load_path}")

    def load_all_adapters(self, base_path: Optional[Union[str, Path]] = None) -> None:
        """
        Load all adapters from a base directory.

        Args:
            base_path: Base directory containing adapter folders. If None, uses the default from config.
        """
        if base_path is None:
            base_path = Path(self.config.paths.checkpoint_dir)
        else:
            base_path = Path(base_path)

        if not base_path.exists():
            raise FileNotFoundError(f"Base path '{base_path}' does not exist")

        # Load each subdirectory as an adapter
        for adapter_dir in base_path.iterdir():
            if adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").exists():
                adapter_name = adapter_dir.name
                if adapter_name not in self.adapters:
                    self.load_adapter(adapter_name, adapter_dir)
