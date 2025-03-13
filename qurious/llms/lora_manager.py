from typing import Dict, Optional

import torch
from peft import (
    LoraConfig as PeftLoraConfig,
)
from peft import (
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
    PreTrainedModel,
    PreTrainedTokenizer,
)

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
        self.model = self._load_base_model()
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

        # Work-around for gemma3 > 1B models
        if "gemma-3" in self.base_name:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self.base_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                attn_implementation="eager",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.base_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
            )
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
        return PeftLoraConfig(**self.lora_config.to_dict())

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
        if config is None:
            config = self._create_peft_config()

        self.model.add_adapter(config, name)

    def get_base_model(self) -> PreTrainedModel:
        """
        Get the base model without any adapters applied.

        Returns:
            The base model
        """
        self.model.disable_adapters()
        return self.model

    def get_model(self, adapter: str) -> PeftModel:
        """
        Get the model with the specified adapter.

        Args:
            adapter: Name of the adapter to use

        Returns:
            The model with the specified adapter applied

        """

        self.model.set_adapter(adapter)
        return self.model
