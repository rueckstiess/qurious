import unittest

from qurious.rl.agents import Agent, QLearningAgent, TabularAgent, ValueBasedAgent
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


class TestAgent(unittest.TestCase):
    """Test the abstract Agent class functionality."""

    def test_agent_is_abstract(self):
        """Test that Agent cannot be instantiated directly."""
        # Should raise TypeError due to abstract methods
        with self.assertRaises(TypeError):
            Agent()

    def test_track_experience_enabled(self):
        """Test enabling experience tracking."""

        # Create a concrete implementation for testing
        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        agent = ConcreteAgent(track_experience=True)
        self.assertIsNotNone(agent.experience)
        self.assertFalse(agent.experience.enable_logging)
        self.assertIsNone(agent.experience.capacity)

        # Test with specific capacity and logging disabled
        agent = ConcreteAgent(track_experience=True, enable_logging=False, capacity=100)
        self.assertIsNotNone(agent.experience)
        self.assertFalse(agent.experience.enable_logging)
        self.assertEqual(agent.experience.capacity, 100)

    def test_track_experience_disabled(self):
        """Test disabling experience tracking."""

        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        agent = ConcreteAgent(track_experience=False)
        self.assertIsNone(agent.experience)

    def test_track_experience_toggle(self):
        """Test toggling experience tracking on and off."""

        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        agent = ConcreteAgent(track_experience=False)
        self.assertIsNone(agent.experience)

        # Enable tracking
        agent.track_experience(True)
        self.assertIsNotNone(agent.experience)

        # Disable tracking
        agent.track_experience(False)
        self.assertIsNone(agent.experience)

    def test_store_experience(self):
        """Test storing transitions in experience buffer."""

        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        # Create agent with experience tracking
        agent = ConcreteAgent(track_experience=True)

        # Store a transition
        agent.store_experience(0, 1, 2.0, 3, False)

        # Check that it was stored
        self.assertEqual(len(agent.experience), 1)
        transition = agent.experience.buffer[0]
        self.assertEqual(transition.state, 0)
        self.assertEqual(transition.action, 1)
        self.assertEqual(transition.reward, 2.0)
        self.assertEqual(transition.next_state, 3)
        self.assertEqual(transition.done, False)

    def test_store_experience_disabled(self):
        """Test that storing experience has no effect when tracking is disabled."""

        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        # Create agent without experience tracking
        agent = ConcreteAgent(track_experience=False)

        # Store a transition (should have no effect)
        agent.store_experience(0, 1, 2.0, 3, False)
        self.assertIsNone(agent.experience)

    def test_experience_capacity(self):
        """Test that experience buffer respects capacity limits."""

        class ConcreteAgent(Agent):
            def choose_action(self, state):
                return 0

            def learn(self, experience):
                pass

            def reset(self):
                pass

        # Create agent with limited capacity
        agent = ConcreteAgent(track_experience=True, capacity=2)

        # Store transitions
        agent.store_experience(0, 0, 1.0, 1, False)
        agent.store_experience(1, 0, 2.0, 2, False)
        agent.store_experience(2, 1, 3.0, 3, False)

        # Should only keep the last 2 transitions
        self.assertEqual(len(agent.experience), 2)
        self.assertEqual(agent.experience.buffer[0].state, 1)
        self.assertEqual(agent.experience.buffer[1].state, 2)


if __name__ == "__main__":
    unittest.main()
