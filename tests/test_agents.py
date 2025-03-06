import unittest

from qurious.rl.agents import QLearningAgent, TabularAgent, ValueBasedAgent
from qurious.rl.policies.policy import DeterministicTabularPolicy, EpsilonGreedyPolicy
from qurious.rl.value_fns import TabularActionValueFunction


class TestTabularAgent(unittest.TestCase):
    def setUp(self):
        """Set up agent for testing."""
        self.n_states = 4
        self.n_actions = 3

        # Create a deterministic policy
        self.policy = DeterministicTabularPolicy(self.n_states, self.n_actions)

        # Create the agent
        self.agent = TabularAgent(self.policy)

    def test_initialization(self):
        """Test if agent is initialized correctly."""
        self.assertEqual(self.agent.policy, self.policy)
        self.assertIsNone(self.agent.value_function)

    def test_choose_action(self):
        """Test choosing actions."""
        # Set specific actions for the policy
        self.policy.policy[0] = 1
        self.policy.policy[1] = 2
        self.policy.policy[2] = 0

        # Agent should return the action from the policy
        self.assertEqual(self.agent.choose_action(0), 1)
        self.assertEqual(self.agent.choose_action(1), 2)
        self.assertEqual(self.agent.choose_action(2), 0)


class TestValueBasedAgent(unittest.TestCase):
    def setUp(self):
        """Set up agent for testing."""
        self.n_states = 3
        self.n_actions = 2
        self.gamma = 0.9

        # Create deterministic policy and value function
        self.policy = DeterministicTabularPolicy(self.n_states, self.n_actions)
        self.value_function = TabularActionValueFunction(self.n_states, self.n_actions)

        # Set initial policy (state 0->action 0, state 1->action 1, state 2->action 0)
        self.policy.policy[0] = 0
        self.policy.policy[1] = 1
        self.policy.policy[2] = 0

        # Set initial value function
        self.value_function.values[0, 0] = 1.0
        self.value_function.values[0, 1] = 0.5
        self.value_function.values[1, 0] = 0.0
        self.value_function.values[1, 1] = 2.0
        self.value_function.values[2, 0] = 1.5
        self.value_function.values[2, 1] = 0.0

        # Create simple value-based agent (uses SARSA updates by default)
        self.agent = ValueBasedAgent(self.policy, self.value_function, self.gamma)

    def test_initialization(self):
        """Test if agent is initialized correctly."""
        self.assertEqual(self.agent.policy, self.policy)
        self.assertEqual(self.agent.value_function, self.value_function)
        self.assertEqual(self.agent.Q, self.value_function)  # Alias should work
        self.assertEqual(self.agent.gamma, self.gamma)

    def test_sarsa_learn(self):
        """Test SARSA learning."""
        # Create transition: (state=0, action=0, reward=1.0, next_state=1, done=False)
        transition = (0, 0, 1.0, 1, False)

        # Before learning
        self.value_function.estimate(0, 0)

        # Learn from transition
        self.agent.learn(transition)

        # After learning:
        # Target = reward + gamma * Q(next_state, next_action)
        # = 1.0 + 0.9 * Q(1, 1) [policy says action 1 for state 1]
        # = 1.0 + 0.9 * 2.0 = 2.8
        # New Q(0, 0) = old Q(0, 0) + alpha * (target - old Q(0, 0))
        # = 1.0 + 0.1 * (2.8 - 1.0) = 1.0 + 0.18 = 1.18
        expected_new_value = 1.0 + 0.1 * (1.0 + 0.9 * 2.0 - 1.0)
        self.assertAlmostEqual(self.value_function.estimate(0, 0), expected_new_value)

    def test_terminal_learn(self):
        """Test learning for terminal state."""
        # Create terminal transition: (state=2, action=0, reward=10.0, next_state=None, done=True)
        transition = (2, 0, 10.0, None, True)

        # Before learning
        initial_value = self.value_function.estimate(2, 0)

        # Learn from transition
        self.agent.learn(transition)

        # For terminal state, target = reward
        expected_new_value = initial_value + 0.1 * (10.0 - initial_value)
        self.assertAlmostEqual(self.value_function.estimate(2, 0), expected_new_value)


class TestQLearningAgent(unittest.TestCase):
    def setUp(self):
        """Set up agent for testing."""
        self.n_states = 3
        self.n_actions = 2
        self.gamma = 0.9

        # Create epsilon-greedy policy and value function
        self.value_function = TabularActionValueFunction(self.n_states, self.n_actions)
        base_policy = DeterministicTabularPolicy(self.n_states, self.n_actions)
        self.policy = EpsilonGreedyPolicy(base_policy, 0.1)

        # Set initial value function
        self.value_function.values[0, 0] = 1.0
        self.value_function.values[0, 1] = 0.5
        self.value_function.values[1, 0] = 0.0
        self.value_function.values[1, 1] = 2.0
        self.value_function.values[2, 0] = 1.5
        self.value_function.values[2, 1] = 0.0

        # Create Q-learning agent
        self.agent = QLearningAgent(self.policy, self.value_function, self.gamma)

    def test_qlearning_learn(self):
        """Test Q-learning update."""
        # Create transition: (state=0, action=0, reward=1.0, next_state=1, done=False)
        transition = (0, 0, 1.0, 1, False)

        # Before learning
        self.value_function.estimate(0, 0)

        # Learn from transition
        self.agent.learn(transition)

        # After learning:
        # Target = reward + gamma * max_a Q(next_state, a)
        # = 1.0 + 0.9 * max(Q(1, 0), Q(1, 1))
        # = 1.0 + 0.9 * max(0.0, 2.0) = 1.0 + 0.9 * 2.0 = 2.8
        # New Q(0, 0) = old Q(0, 0) + alpha * (target - old Q(0, 0))
        # = 1.0 + 0.1 * (2.8 - 1.0) = 1.0 + 0.18 = 1.18
        expected_new_value = 1.0 + 0.1 * (1.0 + 0.9 * 2.0 - 1.0)
        self.assertAlmostEqual(self.value_function.estimate(0, 0), expected_new_value)


if __name__ == "__main__":
    unittest.main()
