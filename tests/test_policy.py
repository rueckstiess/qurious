import unittest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import os
import tempfile

# Import the policy classes
from mini_rl.policy import (
    DeterministicTabularPolicy,
    StochasticTabularPolicy,
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


class TestStochasticTabularPolicy(unittest.TestCase):
    def setUp(self):
        """Set up policy for testing."""
        self.n_states = 4
        self.n_actions = 3
        self.uniform_policy = StochasticTabularPolicy(self.n_states, self.n_actions, initialization="uniform")

        # Set random seed for reproducibility
        np.random.seed(42)
        self.random_policy = StochasticTabularPolicy(self.n_states, self.n_actions, initialization="random")

    def test_uniform_initialization(self):
        """Test uniform initialization."""
        expected = np.ones((self.n_states, self.n_actions)) / self.n_actions
        assert_array_almost_equal(self.uniform_policy.policy, expected)

    def test_random_initialization(self):
        """Test random initialization."""
        # Check that probabilities sum to 1 for each state
        for s in range(self.n_states):
            self.assertAlmostEqual(np.sum(self.random_policy.policy[s]), 1.0)

        # Check that at least some probabilities are different (random)
        self.assertTrue(np.std(self.random_policy.policy) > 0)

    def test_invalid_initialization(self):
        """Test invalid initialization raises error."""
        with self.assertRaises(ValueError):
            StochasticTabularPolicy(self.n_states, self.n_actions, initialization="invalid")

    def test_get_action_probabilities(self):
        """Test getting action probabilities."""
        # For uniform policy, all actions should have equal probability
        for s in range(self.n_states):
            probs = self.uniform_policy.get_action_probabilities(s)
            expected = np.ones(self.n_actions) / self.n_actions
            assert_array_almost_equal(probs, expected)

        # For random policy, should return the stored probabilities
        for s in range(self.n_states):
            probs = self.random_policy.get_action_probabilities(s)
            assert_array_almost_equal(probs, self.random_policy.policy[s])

    def test_get_action(self):
        """Test sampling actions."""
        # This is a probabilistic test
        np.random.seed(42)

        # For uniform policy, sample many actions and check distribution
        state = 0
        n_samples = 10000
        actions = [self.uniform_policy.get_action(state) for _ in range(n_samples)]
        action_counts = np.bincount(actions, minlength=self.n_actions)
        action_probs = action_counts / n_samples

        # Each action should be chosen with approximately equal probability
        expected_probs = np.ones(self.n_actions) / self.n_actions
        assert_array_almost_equal(action_probs, expected_probs, decimal=2)

    def test_update(self):
        """Test updating the policy."""
        state = 1
        action = 2
        value = 10.0

        # Update the policy for state 1, action 2
        self.uniform_policy.update(state, action, value)

        # After update, action 2 should have higher probability for state 1
        probs = self.uniform_policy.get_action_probabilities(state)
        self.assertTrue(probs[action] > 0.9)  # Should be close to 1.0
        self.assertTrue(np.sum(probs) > 0.99)  # Sum should be close to 1.0

    def test_save_and_load(self):
        """Test saving and loading the policy."""
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            filepath = tmp.name

        try:
            self.random_policy.save(filepath)

            # Load the policy
            loaded_policy = StochasticTabularPolicy.load(filepath)

            # Check if loaded policy matches original
            self.assertEqual(loaded_policy.n_states, self.random_policy.n_states)
            self.assertEqual(loaded_policy.n_actions, self.random_policy.n_actions)
            assert_array_almost_equal(loaded_policy.policy, self.random_policy.policy)
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

        # Create a deterministic base policy
        self.base_policy = DeterministicTabularPolicy(self.n_states, self.n_actions)
        self.base_policy.policy[0] = 1  # State 0 -> Action 1
        self.base_policy.policy[1] = 2  # State 1 -> Action 2
        self.base_policy.policy[2] = 3  # State 2 -> Action 3

        # Create epsilon-greedy policy
        self.policy = EpsilonGreedyPolicy(self.base_policy, self.epsilon)

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
