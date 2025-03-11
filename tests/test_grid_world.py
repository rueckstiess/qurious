import unittest

import pytest

from qurious.rl.environments.grid_world import GridWorld, make_grid_world
from qurious.rl.environments.grid_world.utils import extract_actions_from_responses


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

    def test_n_states(self):
        """Test getting the number of states."""
        self.assertEqual(self.env.n_states, 12)  # 4*3 = 12

    def test_n_actions(self):
        """Test getting the number of actions."""
        self.assertEqual(self.env.n_actions, 4)  # UP, RIGHT, DOWN, LEFT

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

    def test_goal_pos_validation(self):
        """Test that goal_pos must be a list of tuples."""
        # Valid goal_pos (list of tuples)
        GridWorld(width=4, height=3, goal_pos=[(1, 2), (3, 2)])

        # Invalid goal_pos - not a tuple
        with self.assertRaises(AssertionError):
            GridWorld(width=4, height=3, goal_pos=[[1, 2]])

        # Invalid goal_pos - tuple of wrong length
        with self.assertRaises(AssertionError):
            GridWorld(width=4, height=3, goal_pos=[(1, 2, 3)])

        # Invalid goal_pos - mixed valid and invalid
        with self.assertRaises(AssertionError):
            GridWorld(width=4, height=3, goal_pos=[(1, 2), [3, 2]])


class TestMakeGridWorld(unittest.TestCase):
    def test_default_parameters(self):
        """Test make_grid_world with default parameters."""
        size = 5
        env = make_grid_world(size)

        # Check basic properties
        self.assertEqual(env.width, size)
        self.assertEqual(env.height, size)
        self.assertIsInstance(env, GridWorld)

        # Check that start_pos and goal_pos are within bounds
        start_row, start_col = env.start_pos
        self.assertTrue(0 <= start_row < size and 0 <= start_col < size)

        # Check that goal_pos is a list with one element
        self.assertEqual(len(env.goal_pos), 1)
        goal_row, goal_col = env.goal_pos[0]
        self.assertTrue(0 <= goal_row < size and 0 <= goal_col < size)

        # Check that start and goal are different
        self.assertNotEqual(env.start_pos, env.goal_pos[0])

        # Check default obstacle ratio
        self.assertIsInstance(env.obstacles, list)
        expected_num_obstacles = int(0.2 * size * size)
        # Allow some variance due to removing obstacles at start/goal
        self.assertLessEqual(len(env.obstacles), expected_num_obstacles)

        # Check step_penalty (there's a typo in the code, should be step_penalty not step_penality)
        self.assertEqual(env.step_penalty, 0.1)

    def test_custom_parameters(self):
        """Test make_grid_world with custom parameters."""
        size = 4
        start_pos = (0, 0)
        goal_pos = [(3, 3)]
        obstacles = 0.1
        terminal_reward = 5.0
        step_penalty = 0.5

        env = make_grid_world(
            size,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            terminal_reward=terminal_reward,
            step_penalty=step_penalty,
        )

        self.assertEqual(env.width, size)
        self.assertEqual(env.height, size)
        self.assertEqual(env.start_pos, start_pos)
        self.assertEqual(env.goal_pos, goal_pos)
        self.assertEqual(env.terminal_reward, terminal_reward)
        self.assertEqual(env.step_penalty, step_penalty)

        # Fix the typo and test again
        env = make_grid_world(
            size,
            start_pos=start_pos,
            goal_pos=goal_pos,
            obstacles=obstacles,
            terminal_reward=terminal_reward,
            step_penalty=step_penalty,  # Using the misspelled parameter name
        )
        self.assertEqual(env.step_penalty, step_penalty)

    def test_different_sizes(self):
        """Test make_grid_world with different grid sizes."""
        for size in [3, 5, 8]:
            env = make_grid_world(size)
            self.assertEqual(env.width, size)
            self.assertEqual(env.height, size)
            self.assertEqual(env.n_states, size * size)

    def test_explicit_obstacles(self):
        """Test make_grid_world with explicitly provided obstacles."""
        size = 4
        obstacles = [(1, 1), (2, 2)]

        env = make_grid_world(size, obstacles=obstacles)

        self.assertEqual(env.obstacles, obstacles)
        self.assertEqual(env.width, size)
        self.assertEqual(env.height, size)

    def test_max_steps_parameter(self):
        """Test make_grid_world with custom max_steps parameter."""
        size = 4
        max_steps = 50

        env = make_grid_world(size, max_steps=max_steps)

        self.assertEqual(env.max_steps, max_steps)

        # Take steps until we exceed max_steps
        env.reset()
        done = False
        for _ in range(max_steps):
            if done:
                break
            _, _, done, _ = env.step(GridWorld.RIGHT)

        # The next step should terminate due to max_steps
        if not done:
            _, _, done, _ = env.step(GridWorld.RIGHT)
            self.assertTrue(done)
            self.assertEqual(env.step_count, max_steps + 1)


class TestExtractActionsFromResponses:
    def test_basic_valid_actions(self):
        """Test extraction of basic valid actions."""
        response = "up down left right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_case_insensitivity(self):
        """Test that the function handles mixed case properly."""
        response = "UP Down LEft RiGHt"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_extra_whitespace(self):
        """Test handling of extra whitespace."""
        response = "  up    down  left    right  "
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_invalid_actions(self):
        """Test filtering of invalid actions."""
        response = "jump up fly down"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down"]
        assert numeric_actions == [0, 2]

    def test_empty_string(self):
        """Test handling of empty string."""
        response = ""
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == []
        assert numeric_actions == []

    def test_no_valid_actions(self):
        """Test handling of input with no valid actions."""
        response = "jump fly run"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == []
        assert numeric_actions == []

    def test_mixed_valid_invalid(self):
        """Test handling of mixed valid and invalid actions."""
        response = "up invalid right nonsense left jump down"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "right", "left", "down"]
        assert numeric_actions == [0, 1, 3, 2]

    def test_comma_separated_actions(self):
        """Test that comma-separated actions are treated as invalid."""
        response = "up, down, left, right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["right"]
        assert numeric_actions == [1]

    def test_non_string_input(self):
        """Test handling of non-string input (which should raise TypeError)."""
        with pytest.raises(AttributeError):
            extract_actions_from_responses(123)

    def test_repeated_actions(self):
        """Test handling of repeated actions."""
        response = "up up down down left right left right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "up", "down", "down", "left", "right", "left", "right"]
        assert numeric_actions == [0, 0, 2, 2, 3, 1, 3, 1]


if __name__ == "__main__":
    unittest.main()
