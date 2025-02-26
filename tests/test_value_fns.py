import unittest
import numpy as np
from numpy.testing import assert_array_equal
import os
import tempfile

# Import the value function classes
from mini_rl.value_fns import (
    TabularStateValueFunction,
    TabularActionValueFunction,
)


class TestTabularStateValueFunction(unittest.TestCase):
    def setUp(self):
        """Set up value function for testing."""
        self.n_states = 5
        self.initial_value = 0.5
        self.value_function = TabularStateValueFunction(self.n_states, self.initial_value)

    def test_initialization(self):
        """Test if the value function is initialized correctly."""
        self.assertEqual(self.value_function.n_states, self.n_states)
        self.assertEqual(self.value_function.initial_value, self.initial_value)
        assert_array_equal(self.value_function.values, np.full(self.n_states, self.initial_value))

    def test_estimate(self):
        """Test estimating values."""
        # By default, all states have the initial value
        for state in range(self.n_states):
            self.assertEqual(self.value_function.estimate(state), self.initial_value)

        # Set specific values for states
        self.value_function.values[1] = 1.0
        self.value_function.values[3] = -0.5

        self.assertEqual(self.value_function.estimate(0), self.initial_value)
        self.assertEqual(self.value_function.estimate(1), 1.0)
        self.assertEqual(self.value_function.estimate(2), self.initial_value)
        self.assertEqual(self.value_function.estimate(3), -0.5)
        self.assertEqual(self.value_function.estimate(4), self.initial_value)

    def test_update(self):
        """Test updating values."""
        state = 2
        target = 1.0
        alpha = 0.5

        # Initial value is 0.5, target is 1.0, alpha is 0.5
        # New value should be 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        self.value_function.update(state, target, alpha)
        self.assertEqual(self.value_function.estimate(state), 0.75)

        # Update again with same target and alpha
        # New value should be 0.75 + 0.5 * (1.0 - 0.75) = 0.875
        self.value_function.update(state, target, alpha)
        self.assertEqual(self.value_function.estimate(state), 0.875)

        # Update with default alpha (0.1)
        state = 3
        target = 0.0
        # New value should be 0.5 + 0.1 * (0.0 - 0.5) = 0.45
        self.value_function.update(state, target)
        self.assertEqual(self.value_function.estimate(state), 0.45)

    def test_reset(self):
        """Test resetting values."""
        # Set specific values for states
        self.value_function.values[1] = 1.0
        self.value_function.values[3] = -0.5

        # Reset
        self.value_function.reset()

        # All values should be initial_value again
        assert_array_equal(self.value_function.values, np.full(self.n_states, self.initial_value))

    def test_save_and_load(self):
        """Test saving and loading the value function."""
        # Set specific values for states
        self.value_function.values[1] = 1.0
        self.value_function.values[3] = -0.5

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            self.value_function.save(filepath)

            # Load the value function
            loaded_value_function = TabularStateValueFunction.load(filepath)

            # Check if loaded value function matches original
            self.assertEqual(loaded_value_function.n_states, self.value_function.n_states)
            self.assertEqual(loaded_value_function.initial_value, self.value_function.initial_value)
            assert_array_equal(loaded_value_function.values, self.value_function.values)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


class TestTabularActionValueFunction(unittest.TestCase):
    def setUp(self):
        """Set up value function for testing."""
        self.n_states = 4
        self.n_actions = 3
        self.initial_value = 0.0
        self.value_function = TabularActionValueFunction(self.n_states, self.n_actions, self.initial_value)

    def test_initialization(self):
        """Test if the value function is initialized correctly."""
        self.assertEqual(self.value_function.n_states, self.n_states)
        self.assertEqual(self.value_function.n_actions, self.n_actions)
        self.assertEqual(self.value_function.initial_value, self.initial_value)
        assert_array_equal(self.value_function.values, np.full((self.n_states, self.n_actions), self.initial_value))

    def test_estimate(self):
        """Test estimating values."""
        # By default, all state-action pairs have the initial value
        for state in range(self.n_states):
            for action in range(self.n_actions):
                self.assertEqual(self.value_function.estimate(state, action), self.initial_value)

        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[2, 1] = -0.5

        self.assertEqual(self.value_function.estimate(1, 0), 1.0)
        self.assertEqual(self.value_function.estimate(2, 1), -0.5)
        self.assertEqual(self.value_function.estimate(0, 0), self.initial_value)

    def test_estimate_all_actions(self):
        """Test estimating values for all actions in a state."""
        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[1, 1] = 0.5
        self.value_function.values[1, 2] = -0.5

        action_values = self.value_function.estimate_all_actions(1)
        expected = np.array([1.0, 0.5, -0.5])
        assert_array_equal(action_values, expected)

    def test_get_best_action(self):
        """Test getting the best action for a state."""
        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[1, 1] = 0.5
        self.value_function.values[1, 2] = -0.5

        best_action = self.value_function.get_best_action(1)
        self.assertEqual(best_action, 0)  # Action 0 has highest value

        # Change values
        self.value_function.values[1, 0] = -1.0
        self.value_function.values[1, 1] = 2.0
        self.value_function.values[1, 2] = 1.5

        best_action = self.value_function.get_best_action(1)
        self.assertEqual(best_action, 1)  # Now action 1 has highest value

    def test_get_best_value(self):
        """Test getting the maximum value for a state."""
        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[1, 1] = 0.5
        self.value_function.values[1, 2] = -0.5

        best_value = self.value_function.get_best_value(1)
        self.assertEqual(best_value, 1.0)  # Max value is 1.0

        # Change values
        self.value_function.values[1, 0] = -1.0
        self.value_function.values[1, 1] = 2.0
        self.value_function.values[1, 2] = 1.5

        best_value = self.value_function.get_best_value(1)
        self.assertEqual(best_value, 2.0)  # Now max value is 2.0

    def test_update(self):
        """Test updating values."""
        state = 2
        action = 1
        target = 1.0
        alpha = 0.5

        # Initial value is 0.0, target is 1.0, alpha is 0.5
        # New value should be 0.0 + 0.5 * (1.0 - 0.0) = 0.5
        self.value_function.update(state, action, target, alpha)
        self.assertEqual(self.value_function.estimate(state, action), 0.5)

        # Update again with same target and alpha
        # New value should be 0.5 + 0.5 * (1.0 - 0.5) = 0.75
        self.value_function.update(state, action, target, alpha)
        self.assertEqual(self.value_function.estimate(state, action), 0.75)

        # Update with default alpha (0.1)
        state = 3
        action = 2
        target = -1.0
        # New value should be 0.0 + 0.1 * (-1.0 - 0.0) = -0.1
        self.value_function.update(state, action, target)
        self.assertEqual(self.value_function.estimate(state, action), -0.1)

    def test_reset(self):
        """Test resetting values."""
        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[2, 1] = -0.5

        # Reset
        self.value_function.reset()

        # All values should be initial_value again
        assert_array_equal(self.value_function.values, np.full((self.n_states, self.n_actions), self.initial_value))

    def test_save_and_load(self):
        """Test saving and loading the value function."""
        # Set specific values for state-action pairs
        self.value_function.values[1, 0] = 1.0
        self.value_function.values[2, 1] = -0.5

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            self.value_function.save(filepath)

            # Load the value function
            loaded_value_function = TabularActionValueFunction.load(filepath)

            # Check if loaded value function matches original
            self.assertEqual(loaded_value_function.n_states, self.value_function.n_states)
            self.assertEqual(loaded_value_function.n_actions, self.value_function.n_actions)
            self.assertEqual(loaded_value_function.initial_value, self.value_function.initial_value)
            assert_array_equal(loaded_value_function.values, self.value_function.values)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
