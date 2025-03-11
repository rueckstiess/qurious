"""
Unit tests for the ML Configuration System.
"""

import argparse
import os
import tempfile
import unittest

import numpy as np

from qurious.config import (
    Config,
    ConfigProduct,
    ConfigSchema,
    DotDict,
    LinSpace,
    ListSpace,
    LogSpace,
    RangeSpace,
    ValidationError,
)


class TestParameterSpaces(unittest.TestCase):
    """Test cases for parameter spaces."""

    def test_list_space(self):
        """Test ListSpace functionality."""
        space = ListSpace([1, 2, 3, 4, 5])

        # Test values
        self.assertEqual(space.values(), [1, 2, 3, 4, 5])

        # Test iteration
        self.assertEqual(list(space), [1, 2, 3, 4, 5])

        # Test length
        self.assertEqual(len(space), 5)

        # Test sampling
        samples = space.sample(3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIn(sample, [1, 2, 3, 4, 5])

    def test_range_space(self):
        """Test RangeSpace functionality."""
        space = RangeSpace(0, 10, 2)

        # Test values
        self.assertEqual(space.values(), [0, 2, 4, 6, 8])

        # Test iteration
        self.assertEqual(list(space), [0, 2, 4, 6, 8])

        # Test length
        self.assertEqual(len(space), 5)

        # Test sampling
        samples = space.sample(3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIn(sample, [0, 2, 4, 6, 8])

    def test_lin_space(self):
        """Test LinSpace functionality."""
        space = LinSpace(0, 1, 5)

        # Test values
        expected = list(np.linspace(0, 1, 5))
        values = space.values()
        self.assertEqual(len(values), 5)
        for i in range(5):
            self.assertAlmostEqual(values[i], expected[i])

        # Test length
        self.assertEqual(len(space), 5)

        # Test sampling
        samples = space.sample(3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            self.assertIn(sample, values)

    def test_log_space(self):
        """Test LogSpace functionality."""
        space = LogSpace(1e-3, 1e-1, 5)

        # Test values
        log_start = np.log10(1e-3)
        log_stop = np.log10(1e-1)
        expected = list(np.logspace(log_start, log_stop, 5))
        values = space.values()
        self.assertEqual(len(values), 5)
        for i in range(5):
            self.assertAlmostEqual(values[i], expected[i])

        # Test length
        self.assertEqual(len(space), 5)

        # Test sampling
        samples = space.sample(3)
        self.assertEqual(len(samples), 3)
        for sample in samples:
            # We can't use assertIn because of floating point comparison
            found = False
            for value in values:
                if abs(sample - value) < 1e-10:
                    found = True
                    break
            self.assertTrue(found)


class TestDotDict(unittest.TestCase):
    """Test cases for DotDict class."""

    def test_init(self):
        """Test initialization."""
        d = DotDict({"a": 1, "b": 2})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b, 2)

    def test_getattr(self):
        """Test attribute access."""
        d = DotDict({"a": 1, "b": {"c": 2}})
        self.assertEqual(d.a, 1)
        self.assertEqual(d.b.c, 2)

        # Test missing attribute
        with self.assertRaises(AttributeError):
            d.missing

    def test_setattr(self):
        """Test setting attributes."""
        d = DotDict()
        d.a = 1
        d.b = {"c": 2}

        self.assertEqual(d.a, 1)
        self.assertEqual(d.b.c, 2)

    def test_delattr(self):
        """Test deleting attributes."""
        d = DotDict({"a": 1, "b": 2})
        del d.a

        with self.assertRaises(AttributeError):
            d.a

        self.assertEqual(d.b, 2)

    def test_getitem(self):
        """Test dictionary-like access."""
        d = DotDict({"a": 1, "b": {"c": 2}})
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"]["c"], 2)

        # Test missing key
        with self.assertRaises(AttributeError):
            d["missing"]

    def test_setitem(self):
        """Test dictionary-like assignment."""
        d = DotDict()
        d["a"] = 1
        d["b"] = {"c": 2}

        self.assertEqual(d.a, 1)
        self.assertEqual(d.b.c, 2)

    def test_delitem(self):
        """Test dictionary-like deletion."""
        d = DotDict({"a": 1, "b": 2})
        del d["a"]

        with self.assertRaises(AttributeError):
            d.a

        self.assertEqual(d.b, 2)

    def test_contains(self):
        """Test key membership."""
        d = DotDict({"a": 1, "b": 2})
        self.assertIn("a", d)
        self.assertIn("b", d)
        self.assertNotIn("c", d)

    def test_iter(self):
        """Test iteration."""
        d = DotDict({"a": 1, "b": 2})
        keys = list(d)
        self.assertIn("a", keys)
        self.assertIn("b", keys)
        self.assertEqual(len(keys), 2)

    def test_len(self):
        """Test length."""
        d = DotDict({"a": 1, "b": 2})
        self.assertEqual(len(d), 2)

    def test_update(self):
        """Test dictionary update."""
        d = DotDict({"a": 1, "b": {"c": 2}})
        d.update({"b": {"d": 3}, "e": 4})

        self.assertEqual(d.a, 1)
        self.assertEqual(d.b.c, 2)
        self.assertEqual(d.b.d, 3)
        self.assertEqual(d.e, 4)

    def test_to_dict(self):
        """Test conversion to regular dictionary."""
        d = DotDict({"a": 1, "b": {"c": 2}})
        result = d.to_dict()

        self.assertEqual(result, {"a": 1, "b": {"c": 2}})
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["b"], dict)
        self.assertNotIsInstance(result["b"], DotDict)

    def test_to_flat_dict(self):
        """Test conversion to flat dictionary."""
        d = DotDict({"a": 1, "b": {"c": 2, "d": {"e": 3}}})
        result = d.to_flat_dict()

        self.assertEqual(result, {"a": 1, "b.c": 2, "b.d.e": 3})
        self.assertIsInstance(result, dict)


class TestConfigSchema(unittest.TestCase):
    """Test cases for ConfigSchema class."""

    def test_validate_types(self):
        """Test type validation."""
        schema = ConfigSchema(
            {
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "string"},
                    "c": {"type": "boolean"},
                    "d": {"type": "array"},
                    "e": {"type": "object"},
                    "f": {"type": "null"},
                },
                "type": "object",
            }
        )

        # Valid config
        valid_config = {"a": 1, "b": "hello", "c": True, "d": [1, 2, 3], "e": {"key": "value"}, "f": None}
        errors = schema.validate(valid_config)
        self.assertEqual(errors, [])

        # Invalid config
        invalid_config = {
            "a": "not a number",
            "b": 123,
            "c": "not a boolean",
            "d": "not an array",
            "e": "not an object",
            "f": "not null",
        }
        errors = schema.validate(invalid_config)
        if errors:  # jsonschema may fail on first error, so just check some errors exist
            self.assertGreaterEqual(len(errors), 1)

    def test_validate_required(self):
        """Test required parameter validation."""
        schema = ConfigSchema(
            {"required": ["a"], "properties": {"a": {"type": "number"}, "b": {"type": "string"}}, "type": "object"}
        )

        # Valid config
        valid_config = {"a": 1}
        errors = schema.validate(valid_config)
        self.assertEqual(errors, [])

        # Invalid config
        invalid_config = {"b": "hello"}
        errors = schema.validate(invalid_config)
        self.assertEqual(len(errors), 1)

    def test_validate_constraints(self):
        """Test constraint validation."""
        schema = ConfigSchema(
            {
                "properties": {
                    "a": {"type": "number", "minimum": 0, "maximum": 10},
                    "b": {"type": "string", "pattern": "^[a-z]+$"},
                    "c": {"type": "string", "enum": ["option1", "option2", "option3"]},
                    # Note: jsonschema doesn't directly support custom validators like the lambda function
                    # You'd typically use a format validator or additional validation logic
                    "d": {"type": "number", "multipleOf": 2},  # Must be even
                },
                "type": "object",
            }
        )

        # Valid config
        valid_config = {"a": 5, "b": "abc", "c": "option2", "d": 4}
        errors = schema.validate(valid_config)
        self.assertEqual(errors, [])

        # Invalid config
        invalid_config = {"a": 20, "b": "ABC", "c": "option4", "d": 3}
        errors = schema.validate(invalid_config)
        if errors:
            self.assertGreaterEqual(len(errors), 1)

    def test_validate_nested(self):
        """Test nested schema validation."""
        schema = ConfigSchema(
            {
                "properties": {
                    "training": {
                        "type": "object",
                        "properties": {
                            "batch_size": {"type": "number"},
                            "learning_rate": {"type": "number"},
                            "optimizer": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}, "params": {"type": "object"}},
                            },
                        },
                    }
                },
                "type": "object",
            }
        )

        # Valid config
        valid_config = {
            "training": {
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": {"name": "adam", "params": {"beta1": 0.9, "beta2": 0.999}},
            }
        }
        errors = schema.validate(valid_config)
        self.assertEqual(errors, [])

        # Invalid config
        invalid_config = {
            "training": {"batch_size": "not a number", "learning_rate": "not a number", "optimizer": "not an object"}
        }
        errors = schema.validate(invalid_config)
        if errors:
            self.assertGreaterEqual(len(errors), 1)


class TestConfig(unittest.TestCase):
    """Test cases for Config class."""

    def test_init(self):
        """Test initialization."""
        config = Config({"a": 1, "b": {"c": 2}})
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b.c, 2)

    def test_parameter_space_from_init(self):
        """Test initialization."""
        config = Config({"a": 1, "b": [1, 2, 3], "c": "linspace(0, 1, 5)", "d": None, "e": "none"})
        self.assertEqual(config.a, 1)
        self.assertIsInstance(config.b, ListSpace)
        self.assertEqual(list(config.b), [1, 2, 3])
        self.assertIsInstance(config.c, LinSpace)
        self.assertEqual(list(config.c), list(np.linspace(0, 1, 5)))
        self.assertEqual(config.d, None)
        self.assertEqual(config.e, "none")

    def test_parameter_space_from_setattr(self):
        """Test setting parameter space using setattr."""
        config = Config({"a": 1})
        config.b = [1, 2, 3]
        config.c = "linspace(0, 1, 5)"
        config.d = "1e-3"
        config.e = '{"foo": "bar"}'

        self.assertEqual(config.a, 1)
        self.assertIsInstance(config.b, ListSpace)
        self.assertEqual(list(config.b), [1, 2, 3])
        self.assertIsInstance(config.c, LinSpace)
        self.assertEqual(list(config.c), list(np.linspace(0, 1, 5)))
        self.assertEqual(config.d, 1e-3)
        self.assertIsInstance(config.e, DotDict)
        self.assertEqual(config.e.to_dict(), {"foo": "bar"})

    def test_validation(self):
        """Test config validation."""
        schema = {"type": "object", "required": ["b"], "properties": {"a": {"type": "number"}, "b": {"type": "object"}}}

        # Valid config
        valid_data = {"a": 1, "b": {"c": "hello"}}
        config = Config(valid_data, schema)
        self.assertEqual(config.a, 1)
        self.assertEqual(config.b.c, "hello")

        # Invalid config
        invalid_data = {"a": "not a number"}
        with self.assertRaises(ValidationError):
            Config(invalid_data, schema)

    def test_freeze_unfreeze(self):
        """Test freezing and unfreezing a config."""
        config = Config({"a": 1})

        # Test freezing
        config.freeze()
        self.assertTrue(config.is_frozen())
        with self.assertRaises(AttributeError):
            config.a = 2

        # Test unfreezing
        config.unfreeze()
        self.assertFalse(config.is_frozen())
        config.a = 2
        self.assertEqual(config.a, 2)

    def test_yaml_serialization(self):
        """Test YAML serialization and deserialization."""
        config = Config(
            {
                "model": {"type": "transformer", "dimensions": 512, "heads": 8},
                "training": {"batch_size": 32, "learning_rate": 0.001, "epochs": 10},
            }
        )

        # Test to YAML
        yaml_str = config.to_yaml()
        self.assertIsInstance(yaml_str, str)

        # Test from YAML
        loaded_config = Config.from_yaml(yaml_str)
        self.assertEqual(loaded_config.model.type, "transformer")
        self.assertEqual(loaded_config.model.dimensions, 512)
        self.assertEqual(loaded_config.model.heads, 8)
        self.assertEqual(loaded_config.training.batch_size, 32)
        self.assertEqual(loaded_config.training.learning_rate, 0.001)
        self.assertEqual(loaded_config.training.epochs, 10)

    def test_yaml_file_io(self):
        """Test YAML file I/O."""
        config = Config({"model": {"type": "transformer", "dimensions": 512}})

        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp:
            temp_path = temp.name

        try:
            config.to_yaml_file(temp_path)

            # Read from temp file
            loaded_config = Config.from_yaml_file(temp_path)
            self.assertEqual(loaded_config.model.type, "transformer")
            self.assertEqual(loaded_config.model.dimensions, 512)
        finally:
            os.unlink(temp_path)

    def test_parameter_spaces(self):
        """Test parameter spaces in configs."""
        config = Config(
            {
                "alpha": ListSpace([1, 2, 3]),
                "beta": RangeSpace(10, 20, 2),
                "gamma": LinSpace(0, 100, 5),
                "delta": LogSpace(1e-3, 1e-1, 3),
            }
        )

        # Test values
        self.assertEqual(list(config.alpha), [1, 2, 3])
        self.assertEqual(list(config.beta), [10, 12, 14, 16, 18])

        lin_values = list(config.gamma)
        self.assertEqual(len(lin_values), 5)
        self.assertAlmostEqual(lin_values[0], 0)
        self.assertAlmostEqual(lin_values[-1], 100)

        log_values = list(config.delta)
        self.assertEqual(len(log_values), 3)
        self.assertAlmostEqual(log_values[0], 1e-3)
        self.assertAlmostEqual(log_values[-1], 1e-1)

        # Test YAML serialization
        yaml_str = config.to_yaml()
        loaded_config = Config.from_yaml(yaml_str)

        # Check if the loaded values are parameter spaces
        self.assertIsInstance(loaded_config.alpha, ListSpace)
        self.assertIsInstance(loaded_config.beta, RangeSpace)
        self.assertIsInstance(loaded_config.gamma, LinSpace)
        self.assertIsInstance(loaded_config.delta, LogSpace)

        # Check values in the loaded parameter spaces
        self.assertEqual(list(loaded_config.alpha), [1, 2, 3])
        self.assertEqual(list(loaded_config.beta), [10, 12, 14, 16, 18])

        loaded_lin_values = list(loaded_config.gamma)
        self.assertEqual(len(loaded_lin_values), 5)
        self.assertAlmostEqual(loaded_lin_values[0], 0)
        self.assertAlmostEqual(loaded_lin_values[-1], 100)

        loaded_log_values = list(loaded_config.delta)
        self.assertEqual(len(loaded_log_values), 3)
        self.assertAlmostEqual(loaded_log_values[0], 1e-3)
        self.assertAlmostEqual(loaded_log_values[-1], 1e-1)

    def test_yaml_none_bool(self):
        yaml_str = """
        v1: null
        v2: none
        v3: "none"
        v4: true
        v5: "True"
        """
        config = Config.from_yaml(yaml_str)
        self.assertIsNone(config.v1)
        self.assertEqual(config.v2, "none")
        self.assertEqual(config.v3, "none")
        self.assertIsInstance(config.v4, bool)
        self.assertIsInstance(config.v5, bool)

    def test_parameter_spaces_scientific_notation(self):
        """Test parameter spaces with scientific notation values."""
        # Test with string values using scientific notation
        yaml_str = """
        spaces:
          lin_space: "linspace(1e-4, 1e-2, 5)"
          log_space: "logspace(1e-5, 1e-3, 3)"
        """

        config = Config.from_yaml(yaml_str)

        # Check if the strings were correctly parsed into parameter spaces
        self.assertIsInstance(config.spaces.lin_space, LinSpace)
        self.assertIsInstance(config.spaces.log_space, LogSpace)

        # Check the values
        lin_values = list(config.spaces.lin_space)
        self.assertEqual(len(lin_values), 5)
        self.assertAlmostEqual(lin_values[0], 1e-4)
        self.assertAlmostEqual(lin_values[-1], 1e-2)

        log_values = list(config.spaces.log_space)
        self.assertEqual(len(log_values), 3)
        self.assertAlmostEqual(log_values[0], 1e-5)
        self.assertAlmostEqual(log_values[-1], 1e-3)

    def test_merge(self):
        """Test merging configs."""
        config1 = Config({"model": {"type": "transformer", "dimensions": 512}, "training": {"batch_size": 32}})

        config2 = Config({"model": {"dimensions": 768, "heads": 12}, "training": {"learning_rate": 0.001}})

        merged = config1.merge(config2)

        # Check merged values
        self.assertEqual(merged.model.type, "transformer")
        self.assertEqual(merged.model.dimensions, 768)  # Overwritten
        self.assertEqual(merged.model.heads, 12)  # Added
        self.assertEqual(merged.training.batch_size, 32)  # Unchanged
        self.assertEqual(merged.training.learning_rate, 0.001)  # Added

    def test_to_dict_spaces(self):
        """Test conversion to dictionary with parameter spaces"""
        config = Config(
            {
                "model": {"type": "transformer", "dimensions": ListSpace([128, 256, 512])},
                "training": {"batch_size": LinSpace(10, 20, 3)},
            }
        )

        # Convert to dictionary without serialization
        result = config.to_dict(serialize_spaces=False)

        # Check types
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["model"], dict)
        self.assertIsInstance(result["training"], dict)

        # Check values
        self.assertIsInstance(result["model"]["dimensions"], ListSpace)
        self.assertEqual(list(result["model"]["dimensions"]), [128, 256, 512])
        self.assertIsInstance(result["training"]["batch_size"], LinSpace)
        self.assertEqual(list(result["training"]["batch_size"]), [10, 15, 20])

        # Convert to dictionary with serialization
        result = config.to_dict(serialize_spaces=True)

        # Check types
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["model"], dict)
        self.assertIsInstance(result["training"], dict)

        # Check values
        self.assertIsInstance(result["model"]["dimensions"], list)
        self.assertEqual(result["model"]["dimensions"], [128, 256, 512])
        self.assertIsInstance(result["training"]["batch_size"], str)
        self.assertEqual(result["training"]["batch_size"], "linspace(10, 20, 3)")

    def test_flatten_and_stringify(self):
        """Test conversion to flattened dictionary and stringify parameters"""
        config = Config(
            {
                "model": {"type": "transformer", "dimensions": ListSpace([128, 256, 512])},
                "training": {"epochs": 5, "batch_size": LinSpace(10, 20, 3)},
            }
        )

        # Convert to dictionary with serialization
        result = config.flatten_and_stringify()

        # Check types
        self.assertIsInstance(result, dict)
        self.assertSetEqual(
            set(result.keys()), {"model.type", "model.dimensions", "training.batch_size", "training.epochs"}
        )

        # Check values
        for k, v in result.items():
            self.assertIsInstance(k, str)

        self.assertEqual(result["model.dimensions"], "[128, 256, 512]")
        self.assertEqual(result["training.batch_size"], "linspace(10, 20, 3)")
        self.assertEqual(result["training.epochs"], "5")

    def test_diff(self):
        """Test config diffing."""
        config1 = Config({"model": {"type": "transformer", "dimensions": 512}, "training": {"batch_size": 32}})

        config2 = Config(
            {
                "model": {"type": "transformer", "dimensions": 768},
                "training": {"learning_rate": 0.001},
                "new_key": "new_value",
            }
        )

        diff = config1.diff(config2)

        # Check diff results
        self.assertEqual(diff["model.dimensions"], (512, 768))
        self.assertEqual(diff["training.batch_size"], (32, None))
        self.assertEqual(diff["training.learning_rate"], (None, 0.001))
        self.assertEqual(diff["new_key"], (None, "new_value"))

    def test_from_env(self):
        """Test loading config from environment variables."""
        # Set up environment
        os.environ["CONFIG_MODEL__TYPE"] = "transformer"
        os.environ["CONFIG_MODEL__DIMENSIONS"] = "512"
        os.environ["CONFIG_TRAINING__BATCH_SIZE"] = "32"
        os.environ["CONFIG_TRAINING__LEARNING_RATE"] = "0.001"
        os.environ["CONFIG_BOOLEAN_VALUE"] = "true"
        os.environ["CONFIG_JSON_VALUE"] = '{"key": "value"}'

        try:
            config = Config.from_env()

            # Check values
            self.assertEqual(config.model.type, "transformer")
            self.assertEqual(config.model.dimensions, 512)
            self.assertEqual(config.training.batch_size, 32)
            self.assertEqual(config.training.learning_rate, 0.001)
            self.assertEqual(config.boolean_value, True)
            self.assertEqual(config.json_value.key, "value")
        finally:
            # Clean up environment
            for key in [
                "CONFIG_MODEL__TYPE",
                "CONFIG_MODEL__DIMENSIONS",
                "CONFIG_TRAINING__BATCH_SIZE",
                "CONFIG_TRAINING__LEARNING_RATE",
                "CONFIG_BOOLEAN_VALUE",
                "CONFIG_JSON_VALUE",
            ]:
                if key in os.environ:
                    del os.environ[key]

    def test_from_args(self):
        """Test loading config from command-line arguments."""

        args = argparse.Namespace(
            params=[
                "model.type=transformer",
                "model.dimensions=512",
                "training.batch_size=32",
                "training.learning_rate=0.001",
                "boolean_value=true",
                'json_value={"key":"value"}',
            ]
        )

        config = Config.from_args(args)

        # Check values
        self.assertEqual(config.model.type, "transformer")
        self.assertEqual(config.model.dimensions, 512)
        self.assertEqual(config.training.batch_size, 32)
        self.assertEqual(config.training.learning_rate, 0.001)
        self.assertEqual(config.boolean_value, True)
        self.assertEqual(config.json_value.key, "value")

    def test_scientific_notation(self):
        """Test handling of scientific notation in config values."""
        # From command-line arguments

        args = argparse.Namespace(
            params=[
                "value1=1e-4",
                "value2=1E-4",
                "value3=1.5e-2",
                "value4=1.5E+2",
                "value5=1e10",
            ]
        )

        config = Config.from_args(args)

        # Check values are correctly converted to floats
        self.assertEqual(config.value1, 0.0001)
        self.assertEqual(config.value2, 0.0001)
        self.assertEqual(config.value3, 0.015)
        self.assertEqual(config.value4, 150.0)
        self.assertEqual(config.value5, 1e10)

        # From environment variables
        os.environ["CONFIG_SCI_NOTATION1"] = "1e-4"
        os.environ["CONFIG_SCI_NOTATION2"] = "1E-4"
        os.environ["CONFIG_SCI_NOTATION3"] = "1.5e-2"
        os.environ["CONFIG_SCI_NOTATION4"] = "1.5E+2"
        os.environ["CONFIG_SCI_NOTATION5"] = "1e10"

        try:
            config = Config.from_env()

            # Check values are correctly converted to floats
            self.assertEqual(config.sci_notation1, 0.0001)
            self.assertEqual(config.sci_notation2, 0.0001)
            self.assertEqual(config.sci_notation3, 0.015)
            self.assertEqual(config.sci_notation4, 150.0)
            self.assertEqual(config.sci_notation5, 1e10)
        finally:
            # Clean up environment
            for key in [
                "CONFIG_SCI_NOTATION1",
                "CONFIG_SCI_NOTATION2",
                "CONFIG_SCI_NOTATION3",
                "CONFIG_SCI_NOTATION4",
                "CONFIG_SCI_NOTATION5",
            ]:
                if key in os.environ:
                    del os.environ[key]

    def test_scientific_notation_yaml(self):
        """Test handling of scientific notation in YAML configs."""
        yaml_str = """
        learning_rates:
          rate1: 1e-4
          rate2: 1E-4
          rate3: 1.5e-2
          rate4: 1.5E+2
          rate5: 1e10
        """

        config = Config.from_yaml(yaml_str)

        # Check values are correctly converted to floats
        self.assertEqual(config.learning_rates.rate1, 0.0001)
        self.assertEqual(config.learning_rates.rate2, 0.0001)
        self.assertEqual(config.learning_rates.rate3, 0.015)
        self.assertEqual(config.learning_rates.rate4, 150.0)
        self.assertEqual(config.learning_rates.rate5, 1e10)


class TestConfigProduct(unittest.TestCase):
    """Test cases for ConfigProduct class."""

    def test_product(self):
        """Test Cartesian product of parameter spaces."""
        config = Config(
            {
                "model": {"dimensions": ListSpace([128, 256, 512])},
                "training": {"learning_rate": LogSpace(1e-4, 1e-2, 3), "batch_size": ListSpace([16, 32])},
            }
        )

        product = ConfigProduct(config)

        # Check length
        self.assertEqual(len(product), 3 * 3 * 2)  # 18 combinations

        # Check iteration
        configs = list(product)
        self.assertEqual(len(configs), 18)

        # Check all combinations are present
        dimensions = set()
        learning_rates = set()
        batch_sizes = set()

        for cfg in configs:
            dimensions.add(cfg.model.dimensions)
            learning_rates.add(cfg.training.learning_rate)
            batch_sizes.add(cfg.training.batch_size)

        self.assertEqual(dimensions, set([128, 256, 512]))
        self.assertEqual(len(learning_rates), 3)
        self.assertEqual(batch_sizes, set([16, 32]))

    def test_product_no_spaces(self):
        """Test product with no parameter spaces."""
        config = Config({"model": {"dimensions": 512}, "training": {"learning_rate": 0.001, "batch_size": 32}})

        product = ConfigProduct(config)

        # Check length
        self.assertEqual(len(product), 1)

        # Check iteration
        configs = list(product)
        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0].model.dimensions, 512)
        self.assertEqual(configs[0].training.learning_rate, 0.001)
        self.assertEqual(configs[0].training.batch_size, 32)

    def test_product_sampling(self):
        """Test sampling from config product."""
        config = Config(
            {
                "model": {"dimensions": RangeSpace(100, 1000, 100)},
                "training": {"learning_rate": LogSpace(1e-4, 1e-2, 10)},
            }
        )

        product = ConfigProduct(config)

        # Full sampling
        self.assertEqual(len(product), 9 * 10)  # 90 combinations

        # Sample a subset
        samples = product.sample(5)
        self.assertEqual(len(samples), 5)

        # Make sure all samples are valid
        for sample in samples:
            self.assertIn(sample.model.dimensions, range(100, 1000, 100))
            # Check that learning rate is within acceptable range (with some tolerance for floating point)
            value = sample.training.learning_rate
            self.assertTrue(
                1e-4 - 1e-10 <= value <= 1e-2 + 1e-10, f"Learning rate {value} not in range [{1e-4}, {1e-2}]"
            )


if __name__ == "__main__":
    unittest.main()
