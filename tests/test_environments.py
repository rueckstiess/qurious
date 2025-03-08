import unittest

from qurious.rl.agents import ExpectedSarsaAgent, QLearningAgent, SarsaAgent
from qurious.rl.environments import Environment
from qurious.rl.environments.grid_world import GridWorld
from qurious.rl.policies import DeterministicTabularPolicy, EpsilonGreedyPolicy


class MockEnvironment(Environment):
    """A mock environment for testing."""

    def __init__(self, n_states=10, n_actions=4):
        super().__init__()
        self._n_states = n_states
        self._n_actions = n_actions

    def step(self, action):
        return 0, 0, False, {}

    def render(self, mode="human"):
        return ""

    @property
    def action_space(self):
        return list(range(self._n_actions))

    @property
    def state_space(self):
        return list(range(self._n_states))

    @property
    def observation_space(self):
        return self.state_space


class TestEnvironmentProperties(unittest.TestCase):
    """Test the Environment class properties."""

    def test_environment_abstract_properties(self):
        """Test that the base Environment properties raise NotImplementedError if not overridden."""
        env = Environment()

        # These should raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            _ = env.action_space

        with self.assertRaises(NotImplementedError):
            _ = env.observation_space

        # Add a mock state_space to verify property implementation
        class MockEnv(Environment):
            @property
            def action_space(self):
                return [0, 1, 2]

            @property
            def state_space(self):
                return [0, 1, 2, 3, 4]

            @property
            def observation_space(self):
                return self.state_space

        mock_env = MockEnv()
        self.assertEqual(mock_env.n_states, 5)
        self.assertEqual(mock_env.n_actions, 3)


class TestGridWorldProperties(unittest.TestCase):
    """Test the GridWorld environment properties."""

    def setUp(self):
        """Set up a grid world for testing."""
        self.grid_world = GridWorld(width=5, height=6, start_pos=(0, 0), goal_pos=[(5, 4)], obstacles=[])

    def test_n_states(self):
        """Test that n_states returns the correct value."""
        # For a 5x6 grid, we should have 30 states
        self.assertEqual(self.grid_world.n_states, 30)

        # Create a new grid with different dimensions
        grid_world_2 = GridWorld(width=3, height=4)
        self.assertEqual(grid_world_2.n_states, 12)

        # Check that the property uses the state_space
        self.assertEqual(self.grid_world.n_states, len(self.grid_world.state_space))

    def test_n_actions(self):
        """Test that n_actions returns the correct value."""
        # GridWorld has 4 actions: UP, RIGHT, DOWN, LEFT
        self.assertEqual(self.grid_world.n_actions, 4)

        # Check that the property uses the action_space
        self.assertEqual(self.grid_world.n_actions, len(self.grid_world.action_space))


class TestAgentFromEnv(unittest.TestCase):
    """Test the from_env class method on agents."""

    def setUp(self):
        """Set up test environment."""
        self.env = MockEnvironment(n_states=20, n_actions=5)

    def test_sarsa_from_env(self):
        """Test creating a SarsaAgent from environment."""
        agent = SarsaAgent.from_env(self.env)

        # Check that components are created with correct dimensions
        self.assertEqual(agent.Q.n_states, 20)
        self.assertEqual(agent.Q.n_actions, 5)

        # Check that policy was created correctly
        self.assertIsInstance(agent.policy, EpsilonGreedyPolicy)
        self.assertEqual(agent.policy.epsilon, 0.5)  # Default epsilon

        # Check base policy dimensions
        self.assertIsInstance(agent.policy.base_policy, DeterministicTabularPolicy)
        self.assertEqual(agent.policy.base_policy.n_states, 20)
        self.assertEqual(agent.policy.base_policy.n_actions, 5)

        # Check agent parameters
        self.assertEqual(agent.gamma, 0.99)  # Default gamma
        self.assertFalse(agent.experience.enable_logging)  # Default logging disabled

    def test_qlearning_from_env(self):
        """Test creating a QLearningAgent from environment."""
        agent = QLearningAgent.from_env(self.env, gamma=0.9, epsilon=0.1)

        # Check custom parameters were passed correctly
        self.assertEqual(agent.gamma, 0.9)
        self.assertEqual(agent.policy.epsilon, 0.1)

        # Check dimensions
        self.assertEqual(agent.Q.n_states, 20)
        self.assertEqual(agent.Q.n_actions, 5)

    def test_expected_sarsa_from_env(self):
        """Test creating an ExpectedSarsaAgent from environment."""
        agent = ExpectedSarsaAgent.from_env(self.env, gamma=0.85, epsilon=0.2, decay_rate=0.95, enable_logging=True)

        # Check custom parameters were passed correctly
        self.assertEqual(agent.gamma, 0.85)
        self.assertEqual(agent.policy.epsilon, 0.2)
        self.assertEqual(agent.policy.decay_rate, 0.95)
        self.assertTrue(agent.experience.enable_logging)

        # Check dimensions
        self.assertEqual(agent.Q.n_states, 20)
        self.assertEqual(agent.Q.n_actions, 5)


if __name__ == "__main__":
    unittest.main()
