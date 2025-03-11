import argparse
import copy
import itertools
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Tuple, TypeVar, Union

import numpy as np
import yaml

from qurious.utils import flatten_dict, process_leaf_values

T = TypeVar("T")


def serialize_value(data: Any) -> Any:
    """Convert parameter spaces to string representations."""
    if isinstance(data, ListSpace):
        return data.values()
    if isinstance(data, RangeSpace):
        return f"range({data.start}, {data.stop}, {data.step})"
    if isinstance(data, LinSpace):
        return f"linspace({data.start}, {data.stop}, {data.num})"
    if isinstance(data, LogSpace):
        return f"logspace({data.start}, {data.stop}, {data.num}, {data.base})"
    return data


def deserialize_value(value: str) -> Any:
    """Parse string representations of parameter spaces. Also handles scientific notation."""

    # Handle regular lists in config
    if isinstance(value, list):
        return ListSpace(value)

    if not isinstance(value, str):
        return value

    if value.startswith("[") and value.endswith("]"):
        # Handle list syntax: [1, 2, 3]
        try:
            items = json.loads(value)
            return ListSpace(items)
        except Exception:
            pass

    # Handle RangeSpace syntax: range(1, 10, 2)
    elif value.startswith("range(") and value.endswith(")"):
        range_str = value[len("range(") : -1]
        try:
            params = [int(p.strip()) for p in range_str.split(",")]
            if len(params) == 2:
                return RangeSpace(params[0], params[1])
            elif len(params) == 3:
                return RangeSpace(params[0], params[1], params[2])
        except Exception:
            pass

    # Handle LinSpace syntax: linspace(0, 1, 5)
    elif value.startswith("linspace(") and value.endswith(")"):
        linspace_str = value[len("linspace(") : -1]
        try:
            # Parse parameters as floats, which handles scientific notation correctly
            parts = [p.strip() for p in linspace_str.split(",")]
            params = [float(p) for p in parts]
            if len(params) == 3:
                return LinSpace(params[0], params[1], int(params[2]))
        except Exception:
            pass

    # Handle LogSpace syntax: logspace(0.001, 0.00001, 5)
    elif value.startswith("logspace(") and value.endswith(")"):
        logspace_str = value[len("logspace(") : -1]
        try:
            parts = [p.strip() for p in logspace_str.split(",")]
            params = [float(p) for p in parts]
            if len(params) == 3:
                return LogSpace(params[0], params[1], int(params[2]))
            elif len(params) == 4:
                return LogSpace(params[0], params[1], int(params[2]), params[3])
        except Exception:
            pass

    elif value.lower() in ["true", "false", "yes", "no"]:
        # Handle boolean values
        return value.lower() in ["true", "yes"]

    # Check if the string might be a scientific notation number
    else:
        if "e" in value.lower() or "." in value.lower():
            try:
                return float(value)
            except Exception:
                pass
        try:
            return json.loads(value)
        except Exception:
            pass

    return value


class ParameterSpace(ABC, Generic[T]):
    """Abstract base class for parameter spaces."""

    @abstractmethod
    def values(self) -> List[T]:
        """Return all values in this space."""
        pass

    @abstractmethod
    def sample(self, n: int = 1, method: str = "uniform") -> List[T]:
        """Sample n values from this space."""
        pass

    def __iter__(self) -> Iterator[T]:
        """Make parameter spaces iterable."""
        return iter(self.values())

    def __len__(self) -> int:
        """Return the number of values in the space."""
        return len(self.values())


class ListSpace(ParameterSpace[T]):
    """A parameter space defined by a list of values."""

    def __init__(self, values: List[T]):
        self._values = list(values)

    def values(self) -> List[T]:
        return self._values

    def sample(self, n: int = 1, method: str = "uniform") -> List[T]:
        import random

        if n >= len(self._values):
            return self._values
        return random.sample(self._values, n)


class RangeSpace(ParameterSpace[int]):
    """A parameter space defined by a range of integers."""

    def __init__(self, start: int, stop: int, step: int = 1):
        self.start = start
        self.stop = stop
        self.step = step
        self._values = list(range(start, stop, step))

    def values(self) -> List[int]:
        return self._values

    def sample(self, n: int = 1, method: str = "uniform") -> List[int]:
        import random

        if n >= len(self._values):
            return self._values
        return random.sample(self._values, n)


class LinSpace(ParameterSpace[float]):
    """A parameter space defined by a linear space."""

    def __init__(self, start: float, stop: float, num: int):
        self.start = start
        self.stop = stop
        self.num = num
        self._values = list(np.linspace(start, stop, num))

    def values(self) -> List[float]:
        return self._values

    def sample(self, n: int = 1, method: str = "uniform") -> List[float]:
        import random

        if n >= len(self._values):
            return self._values
        return random.sample(self._values, n)


class LogSpace(ParameterSpace[float]):
    """A parameter space defined by a logarithmic space."""

    def __init__(self, start: float, stop: float, num: int, base: float = 10.0):
        self.start = start
        self.stop = stop
        self.num = num
        self.base = base

        # Convert to log space
        log_start = np.log(start) / np.log(base)
        log_stop = np.log(stop) / np.log(base)
        self._values = list(np.logspace(log_start, log_stop, num, base=base))

    def values(self) -> List[float]:
        return self._values

    def sample(self, n: int = 1, method: str = "uniform") -> List[float]:
        import random

        if n >= len(self._values):
            return self._values
        return random.sample(self._values, n)


try:
    import jsonschema
    from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False


class ValidationError(Exception):
    """Exception raised for validation errors in the Config class."""

    pass


class ConfigSchema:
    """Schema definition for config validation using jsonschema."""

    def __init__(self, schema: Dict[str, Any]):
        """Initialize a ConfigSchema with a schema definition.

        This class provides a thin wrapper around jsonschema validation.

        Args:
            schema: A JSON Schema compatible schema definition
        """
        self.schema = schema

    def validate(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Validate a config against this schema.

        Args:
            config: The configuration to validate
            path: The path prefix for error messages

        Returns:
            A list of validation error messages, empty if valid
        """
        errors = []
        try:
            jsonschema.validate(instance=config, schema=self.schema)
        except JsonSchemaValidationError as e:
            # Format the error message to include the path
            path_prefix = f"{path}." if path else ""
            error_path = ".".join([str(p) for p in e.path])
            full_path = f"{path_prefix}{error_path}" if error_path else path

            if full_path:
                errors.append(f"Parameter '{full_path}': {e.message}")
            else:
                errors.append(e.message)

        return errors


class DotDict:
    """Dictionary-like class that supports dot notation access."""

    def __init__(self, data: Dict[str, Any] = None):
        """Initialize a DotDict with optional initial data."""
        self._data = {}
        if data:
            self.update(data)

    def __getattr__(self, key: str) -> Any:
        """Get an attribute using dot notation."""
        if key.startswith("_"):
            return super().__getattribute__(key)

        if key not in self._data:
            raise AttributeError(f"No attribute named '{key}'")

        value = self._data[key]
        if isinstance(value, dict) and not isinstance(value, DotDict):
            self._data[key] = DotDict(value)
            return self._data[key]

        return value

    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute using dot notation."""
        if key.startswith("_"):
            super().__setattr__(key, value)
        else:
            self._data[key] = value

    def __delattr__(self, key: str) -> None:
        """Delete an attribute using dot notation."""
        if key.startswith("_"):
            super().__delattr__(key)
        else:
            del self._data[key]

    def __getitem__(self, key: str) -> Any:
        """Get an item using dictionary-like access."""
        return self.__getattr__(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an item using dictionary-like access."""
        self.__setattr__(key, value)

    def __delitem__(self, key: str) -> None:
        """Delete an item using dictionary-like access."""
        self.__delattr__(key)

    def __contains__(self, key: str) -> bool:
        """Check if the dictionary contains a key."""
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys in the dictionary."""
        return iter(self._data)

    def __len__(self) -> int:
        """Get the number of items in the dictionary."""
        return len(self._data)

    def update(self, data: Dict[str, Any]) -> None:
        """Update the dictionary with another dictionary."""
        for key, value in data.items():
            if isinstance(value, dict):
                if key in self._data:
                    if isinstance(self._data[key], DotDict):
                        self._data[key].update(value)
                    else:
                        # Convert dict to DotDict and update
                        self._data[key] = DotDict(value)
                else:
                    # New nested dictionary
                    self._data[key] = DotDict(value)
            else:
                self._data[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a regular dictionary."""
        result = {}

        for key, value in self._data.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert to a flattened dictionary."""
        return flatten_dict(self.to_dict())


class Config(DotDict):
    """Configuration class for ML experiments."""

    def __init__(self, data: Dict[str, Any] = None, schema: Dict[str, Any] = None, frozen: bool = False):
        """Initialize a Config object.

        Args:
            data: Initial configuration data
            schema: Optional schema for validation (can be a jsonschema compatible schema)
            frozen: Whether the config should be immutable
        """
        self._schema = ConfigSchema(schema) if schema else None
        self._frozen = frozen
        super().__init__({})

        if data:
            # Process the data to convert string representations to parameter spaces
            data = process_leaf_values(data, deserialize_value)
            self.update(data)

        if self._schema:
            errors = self.validate()
            if errors:
                raise ValidationError("\n".join(errors))

    def __setattr__(self, key: str, value: Any) -> None:
        """Set an attribute using dot notation."""
        if key.startswith("_"):
            super(DotDict, self).__setattr__(key, value)
        else:
            if getattr(self, "_frozen", False):
                raise AttributeError("Cannot modify a frozen config")
            super().__setattr__(key, deserialize_value(value))

    def validate(self) -> List[str]:
        """Validate the config against its schema."""
        if not self._schema:
            return []

        return self._schema.validate(self.to_dict())

    def freeze(self) -> "Config":
        """Make this config immutable."""
        self._frozen = True
        return self

    def unfreeze(self) -> "Config":
        """Make this config mutable again."""
        self._frozen = False
        return self

    def is_frozen(self) -> bool:
        """Check if this config is frozen."""
        return self._frozen

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "Config":
        """Create a Config from a YAML string."""

        data = yaml.safe_load(yaml_str)

        # Process the data to convert string representations to parameter spaces
        processed_data = process_leaf_values(data, deserialize_value)
        return cls(processed_data)

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "Config":
        """Create a Config from a YAML file."""
        with open(file_path, "r") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """Convert to a YAML string."""

        # Process the data to convert parameter spaces to string representations
        str_data = process_leaf_values(self.to_dict(), serialize_value)
        return yaml.dump(str_data, default_flow_style=False)

    def to_yaml_file(self, file_path: str) -> None:
        """Save to a YAML file."""
        with open(file_path, "w") as f:
            f.write(self.to_yaml())

    def to_dict(self, serialize_spaces: bool = True) -> Dict[str, Any]:
        """Convert to a dictionary."""
        d = super().to_dict()
        if serialize_spaces:
            return process_leaf_values(d, serialize_value)
        return d

    def flatten_and_stringify(self) -> Dict[str, Any]:
        """Flatten the config and convert parameter spaces to string representations."""
        str_dict = process_leaf_values(self.to_dict(), str)
        return flatten_dict(str_dict)

    def merge(self, other: Union[Dict[str, Any], "Config"]) -> "Config":
        """Merge with another config or dictionary."""
        if isinstance(other, Config):
            other = other.to_dict()

        result = Config(self.to_dict())
        result.update(other)

        return result

    def diff(self, other: Union[Dict[str, Any], "Config"]) -> Dict[str, Tuple[Any, Any]]:
        """Get differences between this config and another one."""
        if isinstance(other, Config):
            other = other.to_dict()

        return self._diff_dict(self.to_dict(), other)

    def _diff_dict(self, a: Dict[str, Any], b: Dict[str, Any], path: str = "") -> Dict[str, Tuple[Any, Any]]:
        """Find differences between two dictionaries."""
        result = {}

        # Check keys in a
        for key, value_a in a.items():
            current_path = f"{path}.{key}" if path else key

            if key not in b:
                result[current_path] = (value_a, None)
                continue

            value_b = b[key]

            if isinstance(value_a, dict) and isinstance(value_b, dict):
                nested_diff = self._diff_dict(value_a, value_b, current_path)
                result.update(nested_diff)
            elif (
                isinstance(value_a, ParameterSpace)
                and isinstance(value_b, ParameterSpace)
                and type(value_a) is type(value_b)
                and value_a.values() == value_b.values()
            ):
                continue
            elif value_a != value_b:
                result[current_path] = (value_a, value_b)

        # Check for keys in b that aren't in a
        for key, value_b in b.items():
            current_path = f"{path}.{key}" if path else key

            if key not in a:
                result[current_path] = (None, value_b)

        return result

    @classmethod
    def from_env(cls, prefix: str = "CONFIG_") -> "Config":
        """Create a Config from environment variables."""
        config_data = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                config_keys = config_key.split("__")

                # Parse the value
                parsed_value = deserialize_value(value)

                # Build nested dict structure
                current = config_data
                for i, k in enumerate(config_keys):
                    if i == len(config_keys) - 1:
                        current[k] = parsed_value
                    else:
                        if k not in current:
                            current[k] = {}
                        current = current[k]

        return cls(config_data)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create a Config from command-line arguments."""
        config_data = {}

        # Load from file if specified
        if "config" in args:
            config_data = cls.from_yaml_file(args.config).to_dict()

        # Override with key-value pairs
        if args.params:
            for param in args.params:
                if "=" not in param:
                    continue

                key, value = param.split("=", 1)
                keys = key.split(".")

                # Parse the value
                parsed_value = deserialize_value(value)

                # Build nested dict structure
                current = config_data
                for i, k in enumerate(keys):
                    if i == len(keys) - 1:
                        current[k] = parsed_value
                    else:
                        if k not in current:
                            current[k] = {}
                        current = current[k]

        return cls(config_data)


class ConfigProduct:
    """Class to generate the Cartesian product of parameter spaces in a config."""

    def __init__(self, base_config: Config):
        """Initialize with a base config."""
        self.base_config = base_config
        self._param_spaces = self._find_parameter_spaces(base_config)

    def _find_parameter_spaces(self, config: Config, path: str = "") -> Dict[str, Tuple[str, ParameterSpace]]:
        """Find all parameter spaces in a config."""
        spaces = {}

        for key in config:
            value = config[key]
            current_path = f"{path}.{key}" if path else key

            if isinstance(value, ParameterSpace):
                spaces[current_path] = (current_path, value)
            elif isinstance(value, DotDict):
                nested_spaces = self._find_parameter_spaces(value, current_path)
                spaces.update(nested_spaces)

        return spaces

    def __iter__(self) -> Iterator[Config]:
        """Iterate over all combinations of parameter values."""
        if not self._param_spaces:
            yield copy.deepcopy(self.base_config)
            return

        paths = []
        spaces = []

        for path, (_, space) in self._param_spaces.items():
            paths.append(path)
            spaces.append(list(space))

        for values in itertools.product(*spaces):
            config = copy.deepcopy(self.base_config)

            for i, path in enumerate(paths):
                keys = path.split(".")
                current = config

                for j, key in enumerate(keys):
                    if j == len(keys) - 1:
                        current[key] = values[i]
                    else:
                        current = current[key]

            yield config

    def __len__(self) -> int:
        """Get the total number of configurations."""
        if not self._param_spaces:
            return 1

        count = 1
        for _, (_, space) in self._param_spaces.items():
            count *= len(space)

        return count

    def sample(self, n: int, method: str = "uniform") -> List[Config]:
        """Sample n configurations from the product space."""
        all_configs = list(self)

        import random

        if n >= len(all_configs):
            return all_configs

        return random.sample(all_configs, n)
