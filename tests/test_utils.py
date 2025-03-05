import unittest
from unittest.mock import Mock, patch, MagicMock, call

import os

from qurious.rl.utils import train_agent, run_agent, clear_output


class TestRLUtils(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment
        self.env = Mock()
        self.env.reset.return_value = 0  # Initial state
        
        # Create step responses for a 3-step episode
        self.step_returns = [
            (1, 0.5, False, {}),  # Step 1: Next state 1, reward 0.5, not done
            (2, 1.0, False, {}),  # Step 2: Next state 2, reward 1.0, not done
            (3, 2.0, True, {})    # Step 3: Next state 3, reward 2.0, episode done
        ]
        self.env.step.side_effect = self.step_returns.copy()
        
        # Create mock agent
        self.agent = Mock()
        self.agent.choose_action.return_value = 0  # Always choose action 0
        self.agent.experience = Mock()
        self.agent.experience.get_current_transition.return_value = (0, 0, 0.5, 1, False)
        self.agent.experience.get_current_episode.return_value = [(0, 0, 0.5, 1, False)]
        
        # Create mock callbacks
        self.step_callback = Mock()
        self.episode_callback = Mock()

    def reset_mock_env(self):
        """Reset the environment mock for new tests."""
        self.env.reset.reset_mock()
        self.env.step.reset_mock()
        self.env.step.side_effect = self.step_returns.copy()

    def test_train_agent(self):
        """Test the train_agent function."""
        # Train for a single episode
        train_agent(self.env, self.agent, num_episodes=1, 
                   step_callback=self.step_callback, 
                   episode_callback=self.episode_callback)
        
        # Verify environment interactions
        self.env.reset.assert_called_once()
        self.assertEqual(self.env.step.call_count, 3)
        
        # Verify agent interactions
        self.assertEqual(self.agent.choose_action.call_count, 3)
        self.assertEqual(self.agent.store_experience.call_count, 3)
        self.assertEqual(self.agent.learn.call_count, 3)
        
        # Verify callback invocations
        self.assertEqual(self.step_callback.call_count, 3)
        self.episode_callback.assert_called_once()
        
        # Verify epsilon decay
        self.agent.policy.decay_epsilon.assert_called_once()

    def test_train_agent_multiple_episodes(self):
        """Test training for multiple episodes."""
        # Reset mocks for this test
        self.reset_mock_env()
        self.agent.reset_mock()
        self.agent.policy.reset_mock()
        
        # Make step return the sequence repeatedly
        self.env.step.side_effect = self.step_returns + self.step_returns
        
        train_agent(self.env, self.agent, num_episodes=2)
        
        # Should have called reset twice (once per episode)
        self.assertEqual(self.env.reset.call_count, 2)
        
        # Should have called step 6 times (3 per episode)
        self.assertEqual(self.env.step.call_count, 6)
        
        # Should have called decay_epsilon twice (once per episode)
        self.assertEqual(self.agent.policy.decay_epsilon.call_count, 2)

    def test_train_agent_no_callbacks(self):
        """Test training without callbacks."""
        # Reset mocks for this test
        self.reset_mock_env()
        self.agent.reset_mock()
        self.step_callback.reset_mock()
        self.episode_callback.reset_mock()
        
        train_agent(self.env, self.agent, num_episodes=1)
        
        # Should still perform the core training loop
        self.env.reset.assert_called_once()
        self.assertEqual(self.env.step.call_count, 3)
        self.assertEqual(self.agent.learn.call_count, 3)
        
        # No callbacks should be invoked
        self.step_callback.assert_not_called()
        self.episode_callback.assert_not_called()

    def test_run_agent(self):
        """Test the run_agent function."""
        # Reset mocks for this test
        self.reset_mock_env()
        self.agent.reset_mock()
        self.step_callback.reset_mock()
        self.episode_callback.reset_mock()
        
        # Run for a single episode
        run_agent(self.env, self.agent, num_episodes=1,
                 step_callback=self.step_callback,
                 episode_callback=self.episode_callback)
        
        # Verify environment interactions
        self.env.reset.assert_called_once()
        self.assertEqual(self.env.step.call_count, 3)
        
        # Verify agent interactions
        self.assertEqual(self.agent.choose_action.call_count, 3)
        self.assertEqual(self.agent.store_experience.call_count, 3)
        
        # Unlike train_agent, run_agent should not call learn
        self.agent.learn.assert_not_called()
        
        # Verify callback invocations
        self.assertEqual(self.step_callback.call_count, 3)
        self.episode_callback.assert_called_once()
        
        # Verify epsilon decay is not called
        self.agent.policy.decay_epsilon.assert_not_called()

    def test_run_agent_max_steps(self):
        """Test run_agent with maximum steps per episode."""
        # Reset mocks for this test
        self.reset_mock_env()
        self.agent.reset_mock()
        
        # Override the step method to always return a non-terminal transition
        # Now that step_count is properly incremented, this won't cause an infinite loop
        self.env.step = Mock(return_value=(1, 0.5, False, {}))
        
        # Run with a limit of 2 steps per episode
        run_agent(self.env, self.agent, num_episodes=1, max_steps_per_ep=2)
        
        # Verify we only stepped twice due to the max_steps_per_ep limit
        self.assertEqual(self.env.step.call_count, 2)

    def test_run_agent_multiple_episodes(self):
        """Test running for multiple episodes."""
        # Reset mocks for this test
        self.reset_mock_env()
        self.agent.reset_mock()
        
        # Make step return the sequence repeatedly
        self.env.step.side_effect = self.step_returns + self.step_returns
        
        run_agent(self.env, self.agent, num_episodes=2)
        
        # Should have called reset twice (once per episode)
        self.assertEqual(self.env.reset.call_count, 2)
        
        # Should have called step 6 times (3 per episode)
        self.assertEqual(self.env.step.call_count, 6)

    @patch('os.system')
    @patch('IPython.get_ipython', return_value=None)
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

    @patch('IPython.display.clear_output')
    @patch('IPython.get_ipython')
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

    @patch('os.system')
    @patch('IPython.get_ipython', side_effect=ImportError("No module named IPython"))
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


if __name__ == "__main__":
    unittest.main()