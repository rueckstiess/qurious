import unittest
import numpy as np
from numpy.testing import assert_array_equal

from qurious.mdp import MarkovDecisionProcess


class TestMarkovDecisionProcess(unittest.TestCase):
    def setUp(self):
        """Set up a simple MDP for testing."""
        # Create a simple 3-state, 2-action MDP
        self.states = 3
        self.actions = 2
        self.gamma = 0.9
        self.terminal_states = [2]

        # Initialize transition probabilities
        self.transition_probs = np.zeros((self.states, self.actions, self.states))

        # For state 0, action 0: 80% to state 1, 20% to state 0
        self.transition_probs[0, 0, 1] = 0.8
        self.transition_probs[0, 0, 0] = 0.2

        # For state 0, action 1: 50% to state 1, 50% to state 2
        self.transition_probs[0, 1, 1] = 0.5
        self.transition_probs[0, 1, 2] = 0.5

        # For state 1, action 0: 100% to state 2
        self.transition_probs[1, 0, 2] = 1.0

        # For state 1, action 1: 60% to state 0, 40% to state 1
        self.transition_probs[1, 1, 0] = 0.6
        self.transition_probs[1, 1, 1] = 0.4

        # For state 2 (terminal), all actions lead back to state 2
        self.transition_probs[2, 0, 2] = 1.0
        self.transition_probs[2, 1, 2] = 1.0

        # Initialize rewards
        self.rewards = np.zeros((self.states, self.actions, self.states))

        # Rewards for state 0
        self.rewards[0, 0, 1] = 1.0  # Reward for going from state 0 to 1 with action 0
        self.rewards[0, 1, 2] = 10.0  # Reward for going from state 0 to 2 with action 1

        # Rewards for state 1
        self.rewards[1, 0, 2] = 5.0  # Reward for going from state 1 to 2 with action 0
        self.rewards[1, 1, 0] = -1.0  # Reward for going from state 1 to 0 with action 1

        # Create the MDP
        self.mdp = MarkovDecisionProcess(
            self.states, self.actions, self.transition_probs, self.rewards, self.gamma, self.terminal_states
        )

    def test_initialization(self):
        """Test if the MDP is initialized correctly."""
        self.assertEqual(self.mdp.states, self.states)
        self.assertEqual(self.mdp.actions, self.actions)
        self.assertEqual(self.mdp.gamma, self.gamma)
        self.assertEqual(self.mdp.terminal_states, self.terminal_states)
        assert_array_equal(self.mdp.transition_probs, self.transition_probs)
        assert_array_equal(self.mdp.rewards, self.rewards)

    def test_default_initialization(self):
        """Test initialization with default values."""
        default_mdp = MarkovDecisionProcess(4, 3)
        self.assertEqual(default_mdp.states, 4)
        self.assertEqual(default_mdp.actions, 3)
        self.assertEqual(default_mdp.gamma, 0.99)  # Default gamma
        self.assertEqual(default_mdp.terminal_states, [])  # Default empty list
        expected_probs = np.zeros((4, 3, 4))
        expected_rewards = np.zeros((4, 3, 4))
        assert_array_equal(default_mdp.transition_probs, expected_probs)
        assert_array_equal(default_mdp.rewards, expected_rewards)

    def test_get_transition_prob(self):
        """Test getting transition probabilities."""
        self.assertEqual(self.mdp.get_transition_prob(0, 0, 1), 0.8)
        self.assertEqual(self.mdp.get_transition_prob(0, 1, 2), 0.5)
        self.assertEqual(self.mdp.get_transition_prob(1, 0, 2), 1.0)

    def test_get_reward(self):
        """Test getting rewards."""
        self.assertEqual(self.mdp.get_reward(0, 0, 1), 1.0)
        self.assertEqual(self.mdp.get_reward(0, 1, 2), 10.0)
        self.assertEqual(self.mdp.get_reward(1, 0, 2), 5.0)
        self.assertEqual(self.mdp.get_reward(1, 1, 0), -1.0)

    def test_get_expected_reward(self):
        """Test calculating expected rewards."""
        # Expected reward for state 0, action 0: 0.8 * 1.0 + 0.2 * 0.0 = 0.8
        self.assertAlmostEqual(self.mdp.get_expected_reward(0, 0), 0.8)

        # Expected reward for state 0, action 1: 0.5 * 0.0 + 0.5 * 10.0 = 5.0
        self.assertAlmostEqual(self.mdp.get_expected_reward(0, 1), 5.0)

    def test_is_terminal(self):
        """Test checking if states are terminal."""
        self.assertFalse(self.mdp.is_terminal(0))
        self.assertFalse(self.mdp.is_terminal(1))
        self.assertTrue(self.mdp.is_terminal(2))

    def test_set_transition_prob(self):
        """Test setting transition probabilities."""
        # Change transition prob from state 0, action 0, to state 1
        self.mdp.set_transition_prob(0, 0, 1, 0.7)
        self.assertEqual(self.mdp.get_transition_prob(0, 0, 1), 0.7)

    def test_set_reward(self):
        """Test setting rewards."""
        # Change reward for transition from state 0, action 0, to state 1
        self.mdp.set_reward(0, 0, 1, 2.0)
        self.assertEqual(self.mdp.get_reward(0, 0, 1), 2.0)

    def test_get_next_state_distribution(self):
        """Test getting distribution over next states."""
        dist = self.mdp.get_next_state_distribution(0, 0)
        expected_dist = np.array([0.2, 0.8, 0.0])
        assert_array_equal(dist, expected_dist)

    def test_get_possible_next_states(self):
        """Test getting possible next states."""
        next_states = self.mdp.get_possible_next_states(0, 0)
        self.assertEqual(set(next_states), {0, 1})

        next_states = self.mdp.get_possible_next_states(1, 0)
        self.assertEqual(set(next_states), {2})

    def test_sample_next_state(self):
        """Test sampling next states."""
        # This is a probabilistic test, so we'll sample many times and check statistics
        np.random.seed(42)  # Set seed for reproducibility

        samples = [self.mdp.sample_next_state(0, 0) for _ in range(1000)]
        counts = np.bincount(samples, minlength=self.states)
        probabilities = counts / len(samples)

        # Check that empirical probabilities are close to theoretical ones
        # Allow for some statistical variation
        self.assertAlmostEqual(probabilities[0], 0.2, delta=0.05)
        self.assertAlmostEqual(probabilities[1], 0.8, delta=0.05)
        self.assertAlmostEqual(probabilities[2], 0.0, delta=0.05)

    def test_validation_transition_probs(self):
        """Test validation of transition probabilities."""
        # Create invalid transition probs (don't sum to 1)
        invalid_probs = np.zeros((self.states, self.actions, self.states))
        invalid_probs[0, 0, 1] = 0.7  # Only 0.7 for state 0, action 0

        with self.assertRaises(ValueError):
            MarkovDecisionProcess(self.states, self.actions, invalid_probs, self.rewards)

    def test_validation_shapes(self):
        """Test validation of shape compatibility."""
        # Create transition probs with wrong shape
        wrong_shape_probs = np.zeros((self.states, self.actions + 1, self.states))

        with self.assertRaises(ValueError):
            MarkovDecisionProcess(self.states, self.actions, wrong_shape_probs, self.rewards)

        # Create rewards with wrong shape
        wrong_shape_rewards = np.zeros((self.states, self.actions, self.states + 1))

        with self.assertRaises(ValueError):
            MarkovDecisionProcess(self.states, self.actions, self.transition_probs, wrong_shape_rewards)


if __name__ == "__main__":
    unittest.main()
