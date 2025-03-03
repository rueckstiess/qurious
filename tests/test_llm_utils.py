import unittest

import pytest

from qurious.llms.utils import extract_actions_from_responses


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


if __name__ == "__main__":
    unittest.main()
