import pytest
from qurious.llms.utils import extract_actions_from_responses
import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qurious.llms.utils import run_actions_in_env


class TestExtractActionsFromResponses:
    def test_basic_valid_actions(self):
        """Test extraction of basic valid actions."""
        response = "up, down, left, right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_case_insensitivity(self):
        """Test that the function handles mixed case properly."""
        response = "UP, Down, LEft, RiGHt"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_extra_whitespace(self):
        """Test handling of extra whitespace."""
        response = "  up ,  down,left   , right  "
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "down", "left", "right"]
        assert numeric_actions == [0, 2, 3, 1]

    def test_invalid_actions(self):
        """Test filtering of invalid actions."""
        response = "jump, up, fly, down"
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
        response = "jump, fly, run"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == []
        assert numeric_actions == []

    def test_mixed_valid_invalid(self):
        """Test handling of mixed valid and invalid actions."""
        response = "up, invalid, right, nonsense, left, jump, down"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "right", "left", "down"]
        assert numeric_actions == [0, 1, 3, 2]

    def test_unseparated_actions(self):
        """Test that actions need to be properly comma-separated."""
        response = "up down left right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        # Without commas, it should be treated as a single invalid action
        assert text_actions == []
        assert numeric_actions == []

    def test_non_string_input(self):
        """Test handling of non-string input (which should raise TypeError)."""
        with pytest.raises(AttributeError):
            extract_actions_from_responses(123)

    def test_repeated_actions(self):
        """Test handling of repeated actions."""
        response = "up, up, down, down, left, right, left, right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        assert text_actions == ["up", "up", "down", "down", "left", "right", "left", "right"]
        assert numeric_actions == [0, 0, 2, 2, 3, 1, 3, 1]

    def test_semicolon_separator(self):
        """Test handling of non-comma separators."""
        response = "up; down; left; right"
        text_actions, numeric_actions = extract_actions_from_responses(response)

        # Should interpret the whole string as a single (invalid) action
        assert text_actions == []
        assert numeric_actions == []


class TestRunActionsInEnv(unittest.TestCase):
    def setUp(self):
        # Example data for tests
        self.example1 = {
            "size": 6,
            "start_pos": [0, 2],
            "goal_pos": [2, 4],
            "obstacles": [[0, 1], [3, 2], [1, 5], [0, 1], [3, 1], [2, 5], [0, 5]],
        }

        self.example2 = {
            "size": 7,
            "start_pos": [1, 1],
            "goal_pos": [2, 5],
            "obstacles": [[4, 1], [2, 6], [4, 4], [3, 1], [5, 0], [2, 4], [4, 1], [3, 1]],
        }

        self.example3 = {
            "size": 6,
            "start_pos": [5, 3],
            "goal_pos": [4, 5],
            "obstacles": [[3, 3], [0, 5], [1, 5], [3, 2], [3, 0], [3, 4], [1, 1]],
        }

        # Define some unsuccessful examples
        self.unsuccessful_example = {
            "size": 5,
            "start_pos": [0, 0],
            "goal_pos": [4, 4],
            "obstacles": [],  # No obstacles, but actions won't reach goal
        }

    @patch("qurious.llms.utils.make_env")
    def test_successful_path_example1(self, mock_make_env):
        """Test if correct actions reach the goal in example 1."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"  # Will be used in index_to_state
        mock_env.goal_pos = [[2, 4]]  # This matches what index_to_state will return
        mock_env.index_to_state.return_value = [2, 4]  # Will match goal_pos[0]

        # Set up step returns (_, _, done, _)
        mock_env.step.side_effect = [
            (None, None, False, None),
            (None, None, False, None),
            (None, None, False, None),
            (None, None, True, None),
        ]

        mock_make_env.return_value = mock_env

        # Test correct movement sequence
        numeric_actions = [2, 2, 1, 1]  # down, down, right, right
        result = run_actions_in_env(self.example1, numeric_actions)

        self.assertTrue(result)
        self.assertEqual(mock_env.step.call_count, 4)

    @patch("qurious.llms.utils.make_env")
    def test_successful_path_example2(self, mock_make_env):
        """Test if correct actions reach the goal in example 2."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[2, 5]]
        mock_env.index_to_state.return_value = [2, 5]

        mock_env.step.side_effect = [
            (None, None, False, None),
            (None, None, False, None),
            (None, None, False, None),
            (None, None, False, None),
            (None, None, True, None),
        ]

        mock_make_env.return_value = mock_env

        numeric_actions = [1, 1, 1, 1, 2]  # right, right, right, right, down
        result = run_actions_in_env(self.example2, numeric_actions)

        self.assertTrue(result)
        self.assertEqual(mock_env.step.call_count, 5)

    @patch("qurious.llms.utils.make_env")
    def test_successful_path_example3(self, mock_make_env):
        """Test if correct actions reach the goal in example 3."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[4, 5]]
        mock_env.index_to_state.return_value = [4, 5]

        mock_env.step.side_effect = [(None, None, False, None), (None, None, False, None), (None, None, True, None)]

        mock_make_env.return_value = mock_env

        numeric_actions = [1, 1, 0]  # right, right, up
        result = run_actions_in_env(self.example3, numeric_actions)

        self.assertTrue(result)
        self.assertEqual(mock_env.step.call_count, 3)

    @patch("qurious.llms.utils.make_env")
    def test_unsuccessful_path(self, mock_make_env):
        """Test when actions don't reach the goal."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[4, 4]]
        # Return a position that's not the goal
        mock_env.index_to_state.return_value = [3, 3]

        # No 'done' status reached
        mock_env.step.side_effect = [(None, None, False, None), (None, None, False, None), (None, None, False, None)]

        mock_make_env.return_value = mock_env

        numeric_actions = [0, 1, 2]  # up, right, down (doesn't reach goal)
        result = run_actions_in_env(self.unsuccessful_example, numeric_actions)

        self.assertFalse(result)

    @patch("qurious.llms.utils.make_env")
    def test_early_termination(self, mock_make_env):
        """Test when environment signals done but position doesn't match goal."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[4, 4]]
        # Return a position that's not the goal
        mock_env.index_to_state.return_value = [2, 2]

        # Second action causes early termination
        mock_env.step.side_effect = [(None, None, False, None), (None, None, True, None)]

        mock_make_env.return_value = mock_env

        numeric_actions = [1, 1, 1, 1]  # Would go right 4 times but stops after 2
        result = run_actions_in_env(self.unsuccessful_example, numeric_actions)

        # Should return False as we're done but not at goal
        self.assertFalse(result)
        self.assertEqual(mock_env.step.call_count, 2)  # Should only call step twice

    @patch("qurious.llms.utils.make_env")
    def test_empty_actions(self, mock_make_env):
        """Test with empty actions list."""
        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[4, 4]]
        # If starting position happens to be goal
        mock_env.index_to_state.return_value = [0, 0]

        mock_make_env.return_value = mock_env

        numeric_actions = []  # No actions
        result = run_actions_in_env(self.unsuccessful_example, numeric_actions)

        # Should return False as we're not at the goal
        self.assertFalse(result)
        self.assertEqual(mock_env.step.call_count, 0)  # No steps called

    @patch("qurious.llms.utils.make_env")
    def test_lucky_start_at_goal(self, mock_make_env):
        """Test when agent starts at the goal position."""
        # Set up a special example where start and goal are the same
        lucky_example = {"size": 5, "start_pos": [2, 2], "goal_pos": [2, 2], "obstacles": []}

        # Set up mock environment
        mock_env = MagicMock()
        mock_env.position = "mock_position"
        mock_env.goal_pos = [[2, 2]]
        # Agent is already at goal
        mock_env.index_to_state.return_value = [2, 2]

        mock_make_env.return_value = mock_env

        numeric_actions = []  # No actions needed
        result = run_actions_in_env(lucky_example, numeric_actions)

        # Should return True as we're already at goal
        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
