from pathlib import Path
from typing import Union

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LoraConfig(BaseSettings):
    enabled: bool = True
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__", nested_model_default_partial_update=True)

    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    lora_config: LoraConfig = Field(default_factory=LoraConfig)
    device: str = "auto"


class TrainingConfig(BaseSettings):
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_eval_samples: int = 50
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 1000


class PathConfig(BaseSettings):
    checkpoint_dir: str = "./checkpoints"
    output_dir: str = "./outputs"
    log_dir: str = "./logs"
    data_dir: str = "./data"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QURIOUS_", env_nested_delimiter="__", nested_model_default_partial_update=True
    )

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    paths: PathConfig = PathConfig()

    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> "Config":
        """Load configuration from a YAML file."""
        yaml_file = Path(yaml_file)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_file}")

        with open(yaml_file, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save_yaml(self, yaml_file: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        yaml_file = Path(yaml_file)
        yaml_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and write to yaml
        config_dict = self.model_dump()
        with open(yaml_file, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
