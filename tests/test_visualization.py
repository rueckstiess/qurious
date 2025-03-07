import os
import unittest
from unittest.mock import Mock, patch

from qurious.visualization.utils import clear_output


class TestVisualizationUtils(unittest.TestCase):
    @patch("os.system")
    @patch("IPython.get_ipython", return_value=None)
    def test_clear_output_terminal(self, mock_get_ipython, mock_system):
        """Test clear_output in terminal environment."""
        # Call the function
        result = clear_output()

        # Verify os.system was called with the right command
        if os.name == "nt":
            mock_system.assert_called_once_with("cls")
        else:
            mock_system.assert_called_once_with("clear")

        # Should return False for terminal environment
        self.assertFalse(result)

    @patch("IPython.display.clear_output")
    @patch("IPython.get_ipython")
    def test_clear_output_jupyter(self, mock_get_ipython, mock_jupyter_clear):
        """Test clear_output in Jupyter environment."""
        # Create mock for IPython return value
        mock_ipython = Mock()
        mock_ipython.config = {"IPKernelApp": True}
        mock_get_ipython.return_value = mock_ipython

        # Call the function
        result = clear_output()

        # Verify jupyter_clear was called
        mock_jupyter_clear.assert_called_once_with(wait=True)

        # Should return True for Jupyter environment
        self.assertTrue(result)

    @patch("os.system")
    @patch("IPython.get_ipython", side_effect=ImportError("No module named IPython"))
    def test_clear_output_import_error(self, mock_get_ipython, mock_system):
        """Test clear_output handling ImportError."""
        # Call the function
        result = clear_output()

        # Verify os.system was called for terminal fallback
        if os.name == "nt":
            mock_system.assert_called_once_with("cls")
        else:
            mock_system.assert_called_once_with("clear")

        # Should return False for terminal fallback
        self.assertFalse(result)
