import argparse
import copy
import itertools
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Iterator, List, Tuple, TypeVar, Union

import numpy as np
import yaml

T = TypeVar("T")


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
        If jsonschema is not available, a warning is issued and validation is skipped.

        Args:
            schema: A JSON Schema compatible schema definition
        """
        self.schema = self._convert_schema(schema)

        if not JSONSCHEMA_AVAILABLE:
            import warnings

            warnings.warn(
                "jsonschema package not found. Schema validation will be skipped. "
                "Install jsonschema for full validation support."
            )

    def _convert_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Convert our schema format to jsonschema format if needed."""
        if isinstance(schema, dict) and "type" in schema:
            # This appears to already be in jsonschema format
            return schema

        result = {"type": "object", "properties": {}, "required": [], "additionalProperties": True}

        # Convert each schema key to jsonschema format
        for key, value in schema.items():
            if isinstance(value, dict):
                if "type" in value:
                    # Already has a type specification
                    result["properties"][key] = value.copy()
                    if value.get("required", False):
                        result["required"].append(key)
                        # Remove the required field as it's not part of jsonschema property spec
                        if "required" in result["properties"][key]:
                            del result["properties"][key]["required"]
                else:
                    # Nested schema
                    result["properties"][key] = self._convert_schema(value)
            else:
                # Simple type specification
                result["properties"][key] = {"type": value}

        return result

    def validate(self, config: Dict[str, Any], path: str = "") -> List[str]:
        """Validate a config against this schema.

        Args:
            config: The configuration to validate
            path: The path prefix for error messages

        Returns:
            A list of validation error messages, empty if valid
        """
        if not JSONSCHEMA_AVAILABLE:
            return []  # Skip validation if jsonschema is not available

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
            elif isinstance(value, ParameterSpace):
                # Store parameter spaces as a special format
                if isinstance(value, ListSpace):
                    result[key] = {"__type__": "ListSpace", "values": value.values()}
                elif isinstance(value, RangeSpace):
                    result[key] = {
                        "__type__": "RangeSpace",
                        "start": value.start,
                        "stop": value.stop,
                        "step": value.step,
                    }
                elif isinstance(value, LinSpace):
                    result[key] = {"__type__": "LinSpace", "start": value.start, "stop": value.stop, "num": value.num}
                elif isinstance(value, LogSpace):
                    result[key] = {
                        "__type__": "LogSpace",
                        "start": value.start,
                        "stop": value.stop,
                        "num": value.num,
                        "base": value.base,
                    }
                else:
                    result[key] = list(value.values())
            else:
                result[key] = value

        return result


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
            super().__setattr__(key, value)

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

        def parameter_space_constructor(loader, node):
            data = loader.construct_mapping(node)
            type_name = data.get("__type__")

            if type_name == "ListSpace":
                return ListSpace(data.get("values", []))
            elif type_name == "RangeSpace":
                return RangeSpace(data.get("start", 0), data.get("stop", 0), data.get("step", 1))
            elif type_name == "LinSpace":
                return LinSpace(data.get("start", 0), data.get("stop", 0), data.get("num", 0))
            elif type_name == "LogSpace":
                return LogSpace(data.get("start", 0), data.get("stop", 0), data.get("num", 0), data.get("base", 10.0))

            return data

        def parse_space_dict(data):
            """Parse dictionaries that might represent parameter spaces."""
            if not isinstance(data, dict):
                return data

            if "__type__" in data:
                type_name = data.get("__type__")

                if type_name == "ListSpace":
                    return ListSpace(data.get("values", []))
                elif type_name == "RangeSpace":
                    return RangeSpace(data.get("start", 0), data.get("stop", 0), data.get("step", 1))
                elif type_name == "LinSpace":
                    return LinSpace(data.get("start", 0), data.get("stop", 0), data.get("num", 0))
                elif type_name == "LogSpace":
                    return LogSpace(
                        data.get("start", 0), data.get("stop", 0), data.get("num", 0), data.get("base", 10.0)
                    )

                return data

            # Recursively process nested dictionaries
            return {k: parse_space_dict(v) for k, v in data.items()}

        def parse_string_into_space(value):
            """Parse string representations of parameter spaces."""
            if not isinstance(value, str):
                return value

            # Handle ListSpace syntax: ListSpace([1, 2, 3])
            if value.startswith("ListSpace(") and value.endswith(")"):
                list_str = value[len("ListSpace(") : -1]
                try:
                    # Use eval, but restricted to safe operations for parsing lists
                    # This is safe because we're only using it to parse literal lists
                    import ast

                    list_values = ast.literal_eval(list_str)
                    if isinstance(list_values, list):
                        return ListSpace(list_values)
                except Exception:
                    pass

            # Handle RangeSpace syntax: RangeSpace(1, 10, 2)
            elif value.startswith("RangeSpace(") and value.endswith(")"):
                range_str = value[len("RangeSpace(") : -1]
                try:
                    params = [int(p.strip()) for p in range_str.split(",")]
                    if len(params) == 2:
                        return RangeSpace(params[0], params[1])
                    elif len(params) == 3:
                        return RangeSpace(params[0], params[1], params[2])
                except Exception:
                    pass

            # Handle LinSpace syntax: LinSpace(0, 1, 5)
            elif value.startswith("LinSpace(") and value.endswith(")"):
                linspace_str = value[len("LinSpace(") : -1]
                try:
                    # Parse parameters as floats, which handles scientific notation correctly
                    parts = [p.strip() for p in linspace_str.split(",")]
                    params = [float(p) for p in parts]
                    if len(params) == 3:
                        return LinSpace(params[0], params[1], int(params[2]))
                except Exception:
                    pass

            # Handle LogSpace syntax: LogSpace(0.001, 0.00001, 5)
            elif value.startswith("LogSpace(") and value.endswith(")"):
                logspace_str = value[len("LogSpace(") : -1]
                try:
                    parts = [p.strip() for p in logspace_str.split(",")]
                    params = [float(p) for p in parts]
                    if len(params) == 3:
                        return LogSpace(params[0], params[1], int(params[2]))
                    elif len(params) == 4:
                        return LogSpace(params[0], params[1], int(params[2]), params[3])
                except Exception:
                    pass

            # Check if the string might be a scientific notation number
            elif "e" in value.lower() or "E" in value:
                try:
                    return float(value)
                except Exception:
                    pass

            return value

        def process_config_values(data):
            """Process all values in a config to convert string representations to parameter spaces."""
            if isinstance(data, dict):
                # First check if this dict is a parameter space
                space_dict = parse_space_dict(data)
                # If it's still a dict after parsing, process its values
                if isinstance(space_dict, dict):
                    return {k: process_config_values(v) for k, v in space_dict.items()}
                return space_dict
            elif isinstance(data, list):
                return [process_config_values(item) for item in data]
            else:
                return parse_string_into_space(data)

        # Add our custom constructor for parameter spaces
        yaml.add_constructor("!space", parameter_space_constructor)

        try:
            data = yaml.safe_load(yaml_str)
            # Process the data to convert string representations to parameter spaces
            processed_data = process_config_values(data)
            return cls(processed_data)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML: {e}")

    @classmethod
    def from_yaml_file(cls, file_path: str) -> "Config":
        """Create a Config from a YAML file."""
        with open(file_path, "r") as f:
            return cls.from_yaml(f.read())

    def to_yaml(self) -> str:
        """Convert to a YAML string."""

        def parameter_space_representer(dumper, data):
            if isinstance(data, ListSpace):
                mapping = {"__type__": "ListSpace", "values": data.values()}
            elif isinstance(data, RangeSpace):
                mapping = {"__type__": "RangeSpace", "start": data.start, "stop": data.stop, "step": data.step}
            elif isinstance(data, LinSpace):
                mapping = {"__type__": "LinSpace", "start": data.start, "stop": data.stop, "num": data.num}
            elif isinstance(data, LogSpace):
                mapping = {
                    "__type__": "LogSpace",
                    "start": data.start,
                    "stop": data.stop,
                    "num": data.num,
                    "base": data.base,
                }
            else:
                mapping = {"values": list(data.values())}

            return dumper.represent_mapping("!space", mapping)

        # Register our custom representers
        yaml.add_representer(ListSpace, parameter_space_representer)
        yaml.add_representer(RangeSpace, parameter_space_representer)
        yaml.add_representer(LinSpace, parameter_space_representer)
        yaml.add_representer(LogSpace, parameter_space_representer)

        return yaml.dump(self.to_dict(), default_flow_style=False)

    def to_yaml_file(self, file_path: str) -> None:
        """Save to a YAML file."""
        with open(file_path, "w") as f:
            f.write(self.to_yaml())

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

    @staticmethod
    def _parse_string_value(value: str) -> Any:
        """Parse a string value to the appropriate type.

        This helper function is used to parse string values from environment variables,
        command-line arguments, etc. It handles boolean literals, null values, scientific
        notation, and tries to parse the value as JSON if possible.

        Args:
            value: The string value to parse

        Returns:
            The parsed value with the appropriate type
        """
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        elif value.lower() == "null" or value.lower() == "none":
            return None
        else:
            # Try to parse as JSON
            try:
                return json.loads(value)
            except Exception:
                # Try to parse scientific notation
                try:
                    # Check if the string looks like scientific notation
                    if "e" in value.lower() or "E" in value:
                        return float(value)
                    # Otherwise, keep as string
                    return value
                except Exception:
                    # If all else fails, keep as string
                    return value

    @classmethod
    def from_env(cls, prefix: str = "CONFIG_") -> "Config":
        """Create a Config from environment variables."""
        config_data = {}

        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix) :].lower()
                config_keys = config_key.split("__")

                # Parse the value
                parsed_value = cls._parse_string_value(value)

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
    def from_args(cls, args: List[str] = None) -> "Config":
        """Create a Config from command-line arguments."""
        parser = argparse.ArgumentParser(description="Configuration from command-line arguments")
        parser.add_argument("--config", type=str, help="Path to a YAML config file")
        parser.add_argument("--params", nargs="+", help="Key-value pairs in the format key=value")

        parsed_args = parser.parse_args(args)
        config_data = {}

        # Load from file if specified
        if parsed_args.config:
            config_data = cls.from_yaml_file(parsed_args.config).to_dict()

        # Override with key-value pairs
        if parsed_args.params:
            for param in parsed_args.params:
                if "=" not in param:
                    continue

                key, value = param.split("=", 1)
                keys = key.split(".")

                # Parse the value
                parsed_value = cls._parse_string_value(value)

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
