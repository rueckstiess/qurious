import unittest
from qurious.environments.grid_world import GridWorld


class TestGridWorld(unittest.TestCase):
    def setUp(self):
        """Set up a simple grid world for testing."""
        self.env = GridWorld(
            width=4,
            height=3,
            start_pos=(0, 0),
            goal_pos=[(2, 3)],
            obstacles=[(1, 1)],
            terminal_reward=10.0,
            step_penalty=0.1,
            max_steps=100,
        )

    def test_initialization(self):
        """Test if grid world is initialized correctly."""
        self.assertEqual(self.env.width, 4)
        self.assertEqual(self.env.height, 3)
        self.assertEqual(self.env.start_pos, (0, 0))
        self.assertEqual(self.env.goal_pos, [(2, 3)])
        self.assertEqual(self.env.obstacles, [(1, 1)])
        self.assertEqual(self.env.terminal_reward, 10.0)
        self.assertEqual(self.env.step_penalty, 0.1)
        self.assertEqual(self.env.position, (0, 0))
        self.assertEqual(self.env.step_count, 0)

    def test_reset(self):
        """Test resetting the environment."""
        # Take some steps
        self.env.step(GridWorld.RIGHT)
        self.env.step(GridWorld.DOWN)

        # Reset
        state = self.env.reset()

        # Check if state is correct
        self.assertEqual(self.env.position, (0, 0))
        self.assertEqual(self.env.step_count, 0)
        self.assertEqual(state, 0)  # State index for (0, 0)

    def test_get_state(self):
        """Test getting the state index."""
        # Initial state should be 0
        state = self.env.get_state()
        self.assertEqual(state, 0)

        # Move to (0, 1)
        self.env.step(GridWorld.RIGHT)
        state = self.env.get_state()
        self.assertEqual(state, 1)

        # Move to (1, 1) - should hit obstacle and stay at (0, 1)
        self.env.step(GridWorld.DOWN)
        state = self.env.get_state()
        self.assertEqual(state, 1)

        # Move to (0, 2)
        self.env.step(GridWorld.RIGHT)
        state = self.env.get_state()
        self.assertEqual(state, 2)

    def test_step(self):
        """Test taking steps in the environment."""
        # Test moving right
        next_state, reward, done, info = self.env.step(GridWorld.RIGHT)
        self.assertEqual(self.env.position, (0, 1))
        self.assertEqual(next_state, 1)
        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertEqual(info["step_count"], 1)

        # Test hitting obstacle (try to move down to (1, 1))
        next_state, reward, done, info = self.env.step(GridWorld.DOWN)
        self.assertEqual(self.env.position, (0, 1))  # Should not change
        self.assertEqual(next_state, 1)
        self.assertEqual(reward, -0.1)
        self.assertFalse(done)
        self.assertEqual(info["step_count"], 2)

        # Test reaching goal
        # Move to (0, 2)
        self.env.step(GridWorld.RIGHT)
        # Move to (1, 2)
        self.env.step(GridWorld.DOWN)
        # Move to (2, 2)
        self.env.step(GridWorld.DOWN)
        # Move to (2, 3) - goal
        next_state, reward, done, info = self.env.step(GridWorld.RIGHT)

        self.assertEqual(self.env.position, (2, 3))
        self.assertEqual(next_state, 11)  # 2*4 + 3 = 11
        self.assertEqual(reward, 10.0)
        self.assertTrue(done)
        self.assertEqual(info["step_count"], 6)

    def test_render(self):
        """Test rendering the environment."""
        rendered = self.env.render()
        # Just check if it's a string and not empty
        self.assertIsInstance(rendered, str)
        self.assertTrue(len(rendered) > 0)

        # Check if the agent is at the start position
        first_line = rendered.split("\n")[0]
        self.assertEqual(first_line[0], "A")

    def test_get_num_states(self):
        """Test getting the number of states."""
        self.assertEqual(self.env.get_num_states(), 12)  # 4*3 = 12

    def test_get_num_actions(self):
        """Test getting the number of actions."""
        self.assertEqual(self.env.get_num_actions(), 4)  # UP, RIGHT, DOWN, LEFT

    def test_boundary_checks(self):
        """Test that the agent doesn't go out of bounds."""
        # Try to go up from (0, 0)
        self.env.step(GridWorld.UP)
        self.assertEqual(self.env.position, (0, 0))  # Should not change

        # Try to go left from (0, 0)
        self.env.step(GridWorld.LEFT)
        self.assertEqual(self.env.position, (0, 0))  # Should not change

        # Move to the bottom-right corner (not the goal)
        self.env.position = (2, 2)

        # Try to go down (should stay in bounds)
        self.env.step(GridWorld.DOWN)
        self.assertEqual(self.env.position, (2, 2))  # Should not change

        # Try to go right (should move to goal)
        self.env.step(GridWorld.RIGHT)
        self.assertEqual(self.env.position, (2, 3))

    def test_state_to_index(self):
        """Test converting position to state index."""
        self.assertEqual(self.env.state_to_index((0, 0)), 0)
        self.assertEqual(self.env.state_to_index((0, 3)), 3)
        self.assertEqual(self.env.state_to_index((1, 2)), 6)
        self.assertEqual(self.env.state_to_index((2, 3)), 11)

    def test_index_to_state(self):
        """Test converting state index to position."""
        self.assertEqual(self.env.index_to_state(0), (0, 0))
        self.assertEqual(self.env.index_to_state(3), (0, 3))
        self.assertEqual(self.env.index_to_state(6), (1, 2))
        self.assertEqual(self.env.index_to_state(11), (2, 3))


if __name__ == "__main__":
    unittest.main()
