import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import os
import tempfile

# Import the policy classes
from mini_rl.policy import (
    DeterministicTabularPolicy,
    EpsilonGreedyPolicy,
    SoftmaxPolicy,
)


class TestDeterministicTabularPolicy(unittest.TestCase):
    def setUp(self):
        """Set up policy for testing."""
        self.n_states = 5
        self.n_actions = 3
        self.policy = DeterministicTabularPolicy(self.n_states, self.n_actions)

    def test_initialization(self):
        """Test if policy is initialized correctly."""
        self.assertEqual(self.policy.n_states, self.n_states)
        self.assertEqual(self.policy.n_actions, self.n_actions)
        # Check if all states default to action 0
        assert_array_equal(self.policy.policy, np.zeros(self.n_states, dtype=int))

    def test_initialization_with_default_action(self):
        """Test initialization with custom default action."""
        policy = DeterministicTabularPolicy(self.n_states, self.n_actions, default_action=1)
        assert_array_equal(policy.policy, np.ones(self.n_states, dtype=int))

    def test_get_action(self):
        """Test getting actions."""
        # By default all states map to action 0
        for state in range(self.n_states):
            self.assertEqual(self.policy.get_action(state), 0)

        # Set specific actions for states
        self.policy.policy[1] = 2
        self.policy.policy[3] = 1

        self.assertEqual(self.policy.get_action(0), 0)
        self.assertEqual(self.policy.get_action(1), 2)
        self.assertEqual(self.policy.get_action(2), 0)
        self.assertEqual(self.policy.get_action(3), 1)
        self.assertEqual(self.policy.get_action(4), 0)

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        # Set different actions for different states
        self.policy.policy[0] = 0
        self.policy.policy[1] = 1
        self.policy.policy[2] = 2

        # Action 0 should have probability 1 for state 0
        probs = self.policy.get_action_probabilities(0)
        expected = np.array([1.0, 0.0, 0.0])
        assert_array_equal(probs, expected)

        # Action 1 should have probability 1 for state 1
        probs = self.policy.get_action_probabilities(1)
        expected = np.array([0.0, 1.0, 0.0])
        assert_array_equal(probs, expected)

        # Action 2 should have probability 1 for state 2
        probs = self.policy.get_action_probabilities(2)
        expected = np.array([0.0, 0.0, 1.0])
        assert_array_equal(probs, expected)

    def test_update(self):
        """Test updating the policy."""
        # Update policy for state 2 to use action 1
        self.policy.update(2, 1)
        self.assertEqual(self.policy.get_action(2), 1)

        # Update again to use action 2
        self.policy.update(2, 2)
        self.assertEqual(self.policy.get_action(2), 2)

    def test_save_and_load(self):
        """Test saving and loading the policy."""
        # Set specific actions
        self.policy.policy[1] = 1
        self.policy.policy[3] = 2

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            self.policy.save(filepath)

            # Load the policy
            loaded_policy = DeterministicTabularPolicy.load(filepath)

            # Check if loaded policy matches original
            self.assertEqual(loaded_policy.n_states, self.policy.n_states)
            self.assertEqual(loaded_policy.n_actions, self.policy.n_actions)
            assert_array_equal(loaded_policy.policy, self.policy.policy)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


class TestEpsilonGreedyPolicy(unittest.TestCase):
    def setUp(self):
        """Set up policy for testing."""
        self.n_states = 3
        self.n_actions = 4
        self.epsilon = 0.2
        self.decay_rate = 0.95
        self.min_epsilon = 0.01

        # Create a deterministic base policy
        self.base_policy = DeterministicTabularPolicy(self.n_states, self.n_actions)
        self.base_policy.policy[0] = 1  # State 0 -> Action 1
        self.base_policy.policy[1] = 2  # State 1 -> Action 2
        self.base_policy.policy[2] = 3  # State 2 -> Action 3

        # Create epsilon-greedy policy with decay
        self.policy = EpsilonGreedyPolicy(
            self.base_policy, self.epsilon, decay_rate=self.decay_rate, min_epsilon=self.min_epsilon
        )

    def test_initialization(self):
        """Test if policy is initialized correctly."""
        self.assertEqual(self.policy.n_states, self.n_states)
        self.assertEqual(self.policy.n_actions, self.n_actions)
        self.assertEqual(self.policy.epsilon, self.epsilon)
        self.assertEqual(self.policy.base_policy, self.base_policy)

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        state = 0  # Base policy would choose action 1
        probs = self.policy.get_action_probabilities(state)

        # Probability for action 1 should be (1 - epsilon) + (epsilon / n_actions)
        # Probability for other actions should be epsilon / n_actions
        expected = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        expected[1] += 1 - self.epsilon

        assert_array_almost_equal(probs, expected)

    def test_get_action(self):
        """Test sampling actions."""
        # This is a probabilistic test
        np.random.seed(42)

        state = 0  # Base policy would choose action 1
        n_samples = 10000
        actions = [self.policy.get_action(state) for _ in range(n_samples)]
        action_counts = np.bincount(actions, minlength=self.n_actions)
        action_probs = action_counts / n_samples

        # Expected probabilities from the epsilon-greedy policy
        expected = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        expected[1] += 1 - self.epsilon

        assert_array_almost_equal(action_probs, expected, decimal=2)

    def test_update(self):
        """Test updating the policy."""
        # Update should pass through to the base policy
        state = 0
        action = 3

        self.policy.update(state, action, None)
        self.assertEqual(self.base_policy.get_action(state), action)

        # Test that get_action reflects the update
        np.random.seed(42)
        n_samples = 10000
        actions = [self.policy.get_action(state) for _ in range(n_samples)]
        action_counts = np.bincount(actions, minlength=self.n_actions)
        action_probs = action_counts / n_samples

        # Expected probabilities after update
        expected = np.ones(self.n_actions) * (self.epsilon / self.n_actions)
        expected[3] += 1 - self.epsilon  # Now action 3 is the greedy action

        assert_array_almost_equal(action_probs, expected, decimal=2)

    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        initial_epsilon = self.policy.epsilon

        # Decay once
        new_epsilon = self.policy.decay_epsilon()
        self.assertEqual(new_epsilon, self.policy.epsilon)
        self.assertAlmostEqual(new_epsilon, initial_epsilon * self.decay_rate)

        # Decay multiple times
        for _ in range(10):
            self.policy.decay_epsilon()

        # Epsilon should not go below min_epsilon
        self.assertGreaterEqual(self.policy.epsilon, self.min_epsilon)

    def test_reset_epsilon(self):
        """Test resetting epsilon to initial value."""
        initial_epsilon = self.policy.epsilon

        # Decay a few times
        for _ in range(5):
            self.policy.decay_epsilon()

        # Verify epsilon has changed
        self.assertNotEqual(self.policy.epsilon, initial_epsilon)

        # Reset and verify
        self.policy.reset_epsilon()
        self.assertEqual(self.policy.epsilon, initial_epsilon)

    def test_no_decay_rate(self):
        """Test policy behavior when no decay rate is specified."""
        policy = EpsilonGreedyPolicy(self.base_policy, self.epsilon)
        initial_epsilon = policy.epsilon

        # Decay should have no effect
        policy.decay_epsilon()
        self.assertEqual(policy.epsilon, initial_epsilon)


class TestSoftmaxPolicy(unittest.TestCase):
    def setUp(self):
        """Set up policy for testing."""
        self.n_states = 3
        self.n_actions = 4
        self.temperature = 0.5

        self.policy = SoftmaxPolicy(self.n_states, self.n_actions, self.temperature)

        # Set some action values
        self.policy.action_values[0] = np.array([1.0, 2.0, 0.5, 0.0])
        self.policy.action_values[1] = np.array([0.0, 0.0, 0.0, 0.0])
        self.policy.action_values[2] = np.array([-1.0, -2.0, 3.0, 1.0])

    def test_initialization(self):
        """Test if policy is initialized correctly."""
        # Create a fresh policy for this test
        fresh_policy = SoftmaxPolicy(self.n_states, self.n_actions, self.temperature)

        self.assertEqual(fresh_policy.n_states, self.n_states)
        self.assertEqual(fresh_policy.n_actions, self.n_actions)
        self.assertEqual(fresh_policy.temperature, self.temperature)
        assert_array_equal(fresh_policy.action_values, np.zeros((self.n_states, self.n_actions)))

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        # For state 0, with action values [1.0, 2.0, 0.5, 0.0] and temperature 0.5
        # Probs = softmax(values / temperature)
        state = 0
        probs = self.policy.get_action_probabilities(state)

        # Calculate expected probabilities
        values = self.policy.action_values[state]
        exp_values = np.exp(values / self.temperature)
        expected = exp_values / np.sum(exp_values)

        assert_array_almost_equal(probs, expected)

        # For state 1, all values are 0, so probabilities should be uniform
        state = 1
        probs = self.policy.get_action_probabilities(state)
        expected = np.ones(self.n_actions) / self.n_actions
        assert_array_almost_equal(probs, expected)

    def test_get_action(self):
        """Test sampling actions."""
        # This is a probabilistic test
        np.random.seed(42)

        state = 0
        n_samples = 10000
        actions = [self.policy.get_action(state) for _ in range(n_samples)]
        action_counts = np.bincount(actions, minlength=self.n_actions)
        action_probs = action_counts / n_samples

        # Expected probabilities
        values = self.policy.action_values[state]
        exp_values = np.exp(values / self.temperature)
        expected = exp_values / np.sum(exp_values)

        assert_array_almost_equal(action_probs, expected, decimal=2)

    def test_update(self):
        """Test updating the policy."""
        state = 1
        action = 2
        value = 5.0

        # Update action value
        self.policy.update(state, action, value)
        self.assertEqual(self.policy.action_values[state, action], value)

        # Check that probabilities reflect the update
        probs = self.policy.get_action_probabilities(state)

        # After update, action 2 should have much higher probability
        self.assertTrue(probs[action] > 0.9)  # Should be close to 1.0

    def test_update_from_value_fn(self):
        """Test updating policy from a value function."""

        # Create a mock value function
        class MockValueFunction:
            def estimate_all_actions(self, state):
                # Return different values for each state
                values = {
                    0: np.array([1.0, 2.0, -1.0, 0.0]),
                    1: np.array([0.5, -0.5, 1.5, 0.0]),
                    2: np.array([-1.0, 2.0, 0.0, 1.0]),
                }
                return values[state]

        mock_value_fn = MockValueFunction()

        # Update policy using mock value function
        self.policy.update_from_value_fn(mock_value_fn)

        # Check if action values were updated correctly
        for state in range(self.n_states):
            expected_values = mock_value_fn.estimate_all_actions(state)
            assert_array_almost_equal(
                self.policy.action_values[state], expected_values, err_msg=f"Action values mismatch for state {state}"
            )

    def test_save_and_load(self):
        """Test saving and loading the policy."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            self.policy.save(filepath)

            # Load the policy
            loaded_policy = SoftmaxPolicy.load(filepath)

            # Check if loaded policy matches original
            self.assertEqual(loaded_policy.n_states, self.policy.n_states)
            self.assertEqual(loaded_policy.n_actions, self.policy.n_actions)
            self.assertEqual(loaded_policy.temperature, self.policy.temperature)
            assert_array_almost_equal(loaded_policy.action_values, self.policy.action_values)
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
