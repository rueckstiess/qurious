import unittest

import numpy as np
from numpy.testing import assert_array_equal

from qurious.rl.mdp import MarkovDecisionProcess

# Import the MRP and MDP classes
from qurious.rl.mrp import MarkovRewardProcess


class TestMarkovRewardProcess(unittest.TestCase):
    def setUp(self):
        """Set up a simple MRP for testing."""
        # Create a simple 3-state MRP
        self.states = 3
        self.gamma = 0.9
        self.terminal_states = [2]

        # Initialize transition probabilities
        self.transition_probs = np.zeros((self.states, self.states))

        # For state 0: 70% to state 1, 30% to state 0
        self.transition_probs[0, 1] = 0.7
        self.transition_probs[0, 0] = 0.3

        # For state 1: 60% to state 2, 40% to state 0
        self.transition_probs[1, 2] = 0.6
        self.transition_probs[1, 0] = 0.4

        # For state 2 (terminal): 100% to state 2
        self.transition_probs[2, 2] = 1.0

        # Initialize rewards
        self.rewards = np.zeros((self.states, self.states))

        # Rewards for transitions
        self.rewards[0, 1] = 5.0  # Reward for going from state 0 to 1
        self.rewards[1, 2] = 10.0  # Reward for going from state 1 to 2
        self.rewards[1, 0] = -1.0  # Reward for going from state 1 to 0

        # Create the MRP
        self.mrp = MarkovRewardProcess(
            self.states, self.transition_probs, self.rewards, self.gamma, self.terminal_states
        )

    def test_initialization(self):
        """Test if the MRP is initialized correctly."""
        self.assertEqual(self.mrp.states, self.states)
        self.assertEqual(self.mrp.gamma, self.gamma)
        self.assertEqual(self.mrp.terminal_states, self.terminal_states)
        assert_array_equal(self.mrp.transition_probs, self.transition_probs)
        assert_array_equal(self.mrp.rewards, self.rewards)

    def test_default_initialization(self):
        """Test initialization with default values."""
        default_mrp = MarkovRewardProcess(4)
        self.assertEqual(default_mrp.states, 4)
        self.assertEqual(default_mrp.gamma, 0.99)  # Default gamma
        self.assertEqual(default_mrp.terminal_states, [])  # Default empty list
        expected_probs = np.zeros((4, 4))
        expected_rewards = np.zeros((4, 4))
        assert_array_equal(default_mrp.transition_probs, expected_probs)
        assert_array_equal(default_mrp.rewards, expected_rewards)

    def test_get_transition_prob(self):
        """Test getting transition probabilities."""
        self.assertEqual(self.mrp.get_transition_prob(0, 1), 0.7)
        self.assertEqual(self.mrp.get_transition_prob(1, 2), 0.6)
        self.assertEqual(self.mrp.get_transition_prob(2, 2), 1.0)

    def test_get_reward(self):
        """Test getting rewards."""
        self.assertEqual(self.mrp.get_reward(0, 1), 5.0)
        self.assertEqual(self.mrp.get_reward(1, 2), 10.0)
        self.assertEqual(self.mrp.get_reward(1, 0), -1.0)

    def test_get_expected_reward(self):
        """Test calculating expected rewards."""
        # Expected reward for state 0: 0.7 * 5.0 + 0.3 * 0.0 = 3.5
        self.assertAlmostEqual(self.mrp.get_expected_reward(0), 3.5)

        # Expected reward for state 1: 0.6 * 10.0 + 0.4 * (-1.0) = 5.6
        self.assertAlmostEqual(self.mrp.get_expected_reward(1), 5.6)

    def test_is_terminal(self):
        """Test checking if states are terminal."""
        self.assertFalse(self.mrp.is_terminal(0))
        self.assertFalse(self.mrp.is_terminal(1))
        self.assertTrue(self.mrp.is_terminal(2))

    def test_set_transition_prob(self):
        """Test setting transition probabilities."""
        # Change transition prob from state 0 to state 1
        self.mrp.set_transition_prob(0, 1, 0.8)
        self.assertEqual(self.mrp.get_transition_prob(0, 1), 0.8)

        # Check that expected rewards are updated
        self.assertAlmostEqual(self.mrp.get_expected_reward(0), 4.0)  # 0.8 * 5.0

    def test_set_reward(self):
        """Test setting rewards."""
        # Change reward for transition from state 0 to state 1
        self.mrp.set_reward(0, 1, 2.0)
        self.assertEqual(self.mrp.get_reward(0, 1), 2.0)

        # Check that expected rewards are updated
        self.assertAlmostEqual(self.mrp.get_expected_reward(0), 1.4)  # 0.7 * 2.0

    def test_get_next_state_distribution(self):
        """Test getting distribution over next states."""
        dist = self.mrp.get_next_state_distribution(0)
        expected_dist = np.array([0.3, 0.7, 0.0])
        assert_array_equal(dist, expected_dist)

    def test_get_possible_next_states(self):
        """Test getting possible next states."""
        next_states = self.mrp.get_possible_next_states(0)
        self.assertEqual(set(next_states), {0, 1})

        next_states = self.mrp.get_possible_next_states(1)
        self.assertEqual(set(next_states), {0, 2})

    def test_sample_next_state(self):
        """Test sampling next states."""
        # This is a probabilistic test, so we'll sample many times and check statistics
        np.random.seed(42)  # Set seed for reproducibility

        samples = [self.mrp.sample_next_state(0) for _ in range(1000)]
        counts = np.bincount(samples, minlength=self.states)
        probabilities = counts / len(samples)

        # Check that empirical probabilities are close to theoretical ones
        # Allow for some statistical variation
        self.assertAlmostEqual(probabilities[0], 0.3, delta=0.05)
        self.assertAlmostEqual(probabilities[1], 0.7, delta=0.05)
        self.assertAlmostEqual(probabilities[2], 0.0, delta=0.05)

    def test_validation_transition_probs(self):
        """Test validation of transition probabilities."""
        # Create invalid transition probs (don't sum to 1)
        invalid_probs = np.zeros((self.states, self.states))
        invalid_probs[0, 1] = 0.7  # Only 0.7 for state 0

        with self.assertRaises(ValueError):
            MarkovRewardProcess(self.states, invalid_probs, self.rewards)

    def test_validation_shapes(self):
        """Test validation of shape compatibility."""
        # Create transition probs with wrong shape
        wrong_shape_probs = np.zeros((self.states, self.states + 1))

        with self.assertRaises(ValueError):
            MarkovRewardProcess(self.states, wrong_shape_probs, self.rewards)

        # Create rewards with wrong shape
        wrong_shape_rewards = np.zeros((self.states, self.states + 1))

        with self.assertRaises(ValueError):
            MarkovRewardProcess(self.states, self.transition_probs, wrong_shape_rewards)

    def test_calculate_state_value_function(self):
        """Test calculation of state value function."""
        # Simple MRP where we can calculate the true value function
        # State 0: R(0) = 3.5, transitions to 0 (30%) and 1 (70%)
        # State 1: R(1) = 5.6, transitions to 0 (40%) and 2 (60%)
        # State 2: Terminal state, V(2) = 0

        # Solve the system of equations:
        # V(0) = 3.5 + 0.9 * (0.3 * V(0) + 0.7 * V(1))
        # V(1) = 5.6 + 0.9 * (0.4 * V(0) + 0.6 * V(2))
        # V(2) = 0

        # Simplifying:
        # V(0) = 3.5 + 0.27 * V(0) + 0.63 * V(1)
        # V(1) = 5.6 + 0.36 * V(0) + 0

        # Solving this system:
        # V(0) = (3.5 + 0.63 * (5.6 + 0.36 * V(0))) / (1 - 0.27)
        # V(0) = (3.5 + 3.528 + 0.2268 * V(0)) / 0.73
        # V(0) = (7.028 + 0.2268 * V(0)) / 0.73
        # 0.73 * V(0) - 0.2268 * V(0) = 7.028
        # 0.5032 * V(0) = 7.028
        # V(0) ≈ 13.966

        # Then: V(1) = 5.6 + 0.36 * 13.966 ≈ 10.628

        # Calculate value function using our method
        V = self.mrp.calculate_state_value_function()

        # Check that calculated values match analytical solution
        self.assertAlmostEqual(V[0], 13.966, delta=0.1)
        self.assertAlmostEqual(V[1], 10.628, delta=0.1)
        self.assertAlmostEqual(V[2], 0.0, delta=0.1)

    def test_from_mdp_and_policy(self):
        """Test creating an MRP from an MDP and a policy."""
        # Create a simple MDP
        states = 2
        actions = 2

        # Initialize transition probabilities for MDP
        mdp_transition_probs = np.zeros((states, actions, states))

        # State 0, Action 0: 100% to state 0
        mdp_transition_probs[0, 0, 0] = 1.0

        # State 0, Action 1: 100% to state 1
        mdp_transition_probs[0, 1, 1] = 1.0

        # State 1, Action 0: 100% to state 0
        mdp_transition_probs[1, 0, 0] = 1.0

        # State 1, Action 1: 100% to state 1
        mdp_transition_probs[1, 1, 1] = 1.0

        # Initialize rewards for MDP
        mdp_rewards = np.zeros((states, actions, states))
        mdp_rewards[0, 1, 1] = 10.0  # Reward for going from state 0 to 1 with action 1
        mdp_rewards[1, 0, 0] = 5.0  # Reward for going from state 1 to 0 with action 0

        # Create the MDP
        mdp = MarkovDecisionProcess(states, actions, mdp_transition_probs, mdp_rewards)

        # Create a policy: 50% action 0, 50% action 1 in both states
        policy = np.ones((states, actions)) * 0.5

        # Create an MRP from the MDP and policy
        mrp = MarkovRewardProcess.from_mdp_and_policy(mdp, policy)

        # Check MRP transition probabilities
        # State 0: 50% to state 0 (from action 0), 50% to state 1 (from action 1)
        self.assertAlmostEqual(mrp.get_transition_prob(0, 0), 0.5)
        self.assertAlmostEqual(mrp.get_transition_prob(0, 1), 0.5)

        # State 1: 50% to state 0 (from action 0), 50% to state 1 (from action 1)
        self.assertAlmostEqual(mrp.get_transition_prob(1, 0), 0.5)
        self.assertAlmostEqual(mrp.get_transition_prob(1, 1), 0.5)

        # Check MRP rewards
        # R(0,1) should be 10.0 (from mdp_rewards[0,1,1])
        self.assertAlmostEqual(mrp.get_reward(0, 1), 10.0)

        # R(1,0) should be 5.0 (from mdp_rewards[1,0,0])
        self.assertAlmostEqual(mrp.get_reward(1, 0), 5.0)

        # Check expected rewards
        # For state 0: 0.5 * 0 (to state 0) + 0.5 * 10 (to state 1) = 5.0
        self.assertAlmostEqual(mrp.get_expected_reward(0), 5.0)

        # For state 1: 0.5 * 5 (to state 0) + 0.5 * 0 (to state 1) = 2.5
        self.assertAlmostEqual(mrp.get_expected_reward(1), 2.5)


if __name__ == "__main__":
    unittest.main()
