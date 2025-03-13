import unittest
from io import StringIO

import numpy as np
from loguru import logger

from qurious.rl.experience import Experience, Transition


class TestExperience(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.empty_exp = Experience(capacity=100)

        # Create single episode experience
        self.single_episode_exp = Experience(capacity=100)
        transitions = [
            Transition(0, 1, 0.5, 1, False),
            Transition(1, 0, 1.0, 2, False),
            Transition(2, 1, -1.0, 3, True),
        ]
        for t in transitions:
            self.single_episode_exp.add(t)

        # Create multi-episode experience
        self.multi_episode_exp = Experience(capacity=100)
        # Episode 1
        self.multi_episode_exp.add(Transition(0, 1, 0.5, 1, False))
        self.multi_episode_exp.add(Transition(1, 0, 1.0, 2, True))
        # Episode 2
        self.multi_episode_exp.add(Transition(0, 0, -0.5, 1, False))
        self.multi_episode_exp.add(Transition(1, 1, 0.0, 2, False))
        self.multi_episode_exp.add(Transition(2, 0, 2.0, 3, True))

    def test_init(self):
        """Test Experience initialization."""
        exp = Experience(capacity=10)
        self.assertEqual(exp.capacity, 10)
        self.assertEqual(len(exp.buffer), 0)
        self.assertEqual(len(exp.episode_boundaries), 0)
        self.assertEqual(len(exp._current_episode), 0)

        # Test unlimited capacity
        exp = Experience()
        self.assertIsNone(exp.capacity)

    def test_add_transition(self):
        """Test adding transitions to experience buffer."""
        t = Transition(0, 1, 0.5, 1, False)
        self.empty_exp.add(t)

        self.assertEqual(len(self.empty_exp.buffer), 1)
        self.assertEqual(len(self.empty_exp._current_episode), 1)
        self.assertEqual(len(self.empty_exp.episode_boundaries), 0)
        self.assertEqual(self.empty_exp.buffer[0], t)

    def test_add_episode_boundary(self):
        """Test handling of episode boundaries."""
        exp = Experience(capacity=100)
        exp.add(Transition(0, 1, 0.5, 1, False))
        exp.add(Transition(1, 0, 1.0, 2, True))  # Episode end

        self.assertEqual(len(exp.episode_boundaries), 1)
        self.assertEqual(exp.episode_boundaries[0], 1)
        self.assertEqual(len(exp._current_episode), 0)

    def test_capacity_limit(self):
        """Test that capacity limit is enforced."""
        exp = Experience(capacity=2)
        exp.add(Transition(0, 1, 0.5, 1, False))
        exp.add(Transition(1, 0, 1.0, 2, False))
        exp.add(Transition(2, 1, -1.0, 3, False))

        self.assertEqual(len(exp.buffer), 2)
        self.assertEqual(exp.buffer[0].state, 1)  # First transition should be evicted

    def test_sample_batch(self):
        """Test batch sampling."""
        batch = self.multi_episode_exp.sample_batch(batch_size=2)

        self.assertEqual(len(batch), 2)
        for t in batch:
            self.assertIsInstance(t, Transition)

    def test_sample_batch_with_replacement(self):
        """Test that batch sampling is done with replacement."""
        np.random.seed(42)  # For reproducibility
        batch1 = self.multi_episode_exp.sample_batch(batch_size=10)
        batch2 = self.multi_episode_exp.sample_batch(batch_size=10)

        self.assertEqual(len(batch1), 10)
        self.assertEqual(len(batch2), 10)
        self.assertTrue(any(t1 != t2 for t1, t2 in zip(batch1, batch2)))

    def test_sample_episode(self):
        """Test episode sampling."""
        episode = self.multi_episode_exp.sample_episode()

        self.assertTrue(len(episode) > 0)
        self.assertTrue(episode[-1].done)  # Last transition should be terminal
        for t in episode:
            self.assertIsInstance(t, Transition)

    def test_sample_episode_empty(self):
        """Test episode sampling from empty buffer."""
        episode = self.empty_exp.sample_episode()
        self.assertEqual(len(episode), 0)

    def test_get_current_transition(self):
        """Test current transition access."""
        self.single_episode_exp.add(Transition(2, 1, 0.3, 0, False))
        transaction = Transition(0, 1, 0.5, 1, False)
        self.single_episode_exp.add(transaction)

        current = self.single_episode_exp.get_current_transition()
        self.assertIsNotNone(current)
        self.assertEqual(current, transaction)

    def test_get_current_transition_empty(self):
        """Test current transition access from empty buffer."""
        current = self.empty_exp.get_current_transition()
        self.assertIsNone(current)

    def test_get_current_episode(self):
        """Test current episode access."""
        self.single_episode_exp.add(Transition(0, 1, 0.5, 1, False))  # Start new episode

        current = self.single_episode_exp.get_current_episode()
        self.assertEqual(len(current), 1)
        self.assertFalse(current[0].done)

    def test_get_current_episode_after_done(self):
        """Test that get_current_episode returns the last completed episode after done."""
        exp = Experience()
        # Add a complete episode
        transitions = [
            Transition(0, 1, 0.5, 1, False),
            Transition(1, 0, 1.0, 2, False),
            Transition(2, 1, -1.0, 3, True),  # Episode end
        ]
        for t in transitions:
            exp.add(t)

        # After the done transition, get_current_episode should return the completed episode
        current = exp.get_current_episode()
        self.assertEqual(len(current), 3)
        self.assertTrue(current[-1].done)

        # Once we add a new transition, get_current_episode should return only that new transition
        exp.add(Transition(3, 0, 0.5, 4, False))
        current = exp.get_current_episode()
        self.assertEqual(len(current), 1)
        self.assertEqual(current[0].state, 3)

    def test_clear(self):
        """Test clearing experience buffer."""
        self.multi_episode_exp.clear()

        self.assertEqual(len(self.multi_episode_exp.buffer), 0)
        self.assertEqual(len(self.multi_episode_exp.episode_boundaries), 0)
        self.assertEqual(len(self.multi_episode_exp._current_episode), 0)

    def test_size_property(self):
        """Test size property."""
        self.assertEqual(self.multi_episode_exp.size, 5)

    def test_episode_boundary_consistency(self):
        """Test consistency of episode boundaries."""
        # All episodes should end with done=True
        for idx in self.multi_episode_exp.episode_boundaries:
            self.assertTrue(self.multi_episode_exp.buffer[idx].done)

        # Episodes should be in ascending order
        boundaries = self.multi_episode_exp.episode_boundaries
        for i in range(len(boundaries) - 1):
            self.assertLess(boundaries[i], boundaries[i + 1])

    def test_transition_iteration(self):
        exp = Experience()

        # Create some test transitions
        transitions = [
            Transition(state=1, action=0, reward=1.0, next_state=2, done=False),
            Transition(state=2, action=1, reward=-1.0, next_state=3, done=False),
            Transition(state=3, action=0, reward=2.0, next_state=4, done=True),
        ]

        # Add transitions to experience
        for t in transitions:
            exp.add(t)

        # Test iteration over transitions
        collected = list(exp)
        self.assertEqual(len(collected), 3)
        for a, b in zip(collected, transitions):
            self.assertEqual(a, b)

    def test_episode_iteration(self):
        exp = Experience()

        # Create two episodes
        episode1 = [
            Transition(state=1, action=0, reward=1.0, next_state=2, done=False),
            Transition(state=2, action=1, reward=-1.0, next_state=3, done=True),
        ]

        episode2 = [
            Transition(state=4, action=0, reward=2.0, next_state=5, done=False),
            Transition(state=5, action=1, reward=3.0, next_state=6, done=True),
        ]

        # Add all transitions
        for t in episode1 + episode2:
            exp.add(t)

        # Test iteration over episodes
        episodes = list(exp.iter_episodes())
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0], episode1)
        self.assertEqual(episodes[1], episode2)

    def test_episode_iteration_with_incomplete(self):
        exp = Experience()

        # Create one complete episode and one incomplete
        complete_episode = [
            Transition(state=1, action=0, reward=1.0, next_state=2, done=False),
            Transition(state=2, action=1, reward=-1.0, next_state=3, done=True),
        ]

        incomplete_episode = [
            Transition(state=4, action=0, reward=2.0, next_state=5, done=False),
            Transition(state=5, action=1, reward=3.0, next_state=6, done=False),
        ]

        # Add all transitions
        for t in complete_episode:
            exp.add(t)
        for t in incomplete_episode:
            exp.add(t)

        # Test iteration over episodes
        episodes = list(exp.iter_episodes())
        self.assertEqual(len(episodes), 2)
        self.assertEqual(episodes[0], complete_episode)
        self.assertEqual(episodes[1], incomplete_episode)

    def test_empty_experience_iteration(self):
        exp = Experience()

        # Test iteration over empty experience
        self.assertEqual(list(exp), [])
        self.assertEqual(list(exp.iter_episodes()), [])

    def test_logging_disabled_by_default(self):
        """Test that logging is disabled by default."""
        exp = Experience(capacity=10)
        self.assertFalse(exp.enable_logging)

    def test_logging_enabled(self):
        """Test that logging can be enabled."""
        exp = Experience(capacity=10, enable_logging=True)
        self.assertTrue(exp.enable_logging)

    def test_logging_add_transition(self):
        """Test that transitions are logged when logging is enabled."""
        # Setup a string IO as log handler to capture logs
        log_capture = StringIO()
        logger.remove()  # Remove default handlers
        logger_id = logger.add(log_capture, level="DEBUG")

        # Create experience with logging enabled
        exp = Experience(capacity=10, enable_logging=True)

        # Add a transition
        t = Transition(state=1, action=2, reward=0.5, next_state=3, done=False)
        exp.add(t)

        # Check log output
        log_content = log_capture.getvalue()
        self.assertIn("Added transition:", log_content)
        self.assertIn("state=1", log_content)
        self.assertIn("action=2", log_content)
        self.assertIn("reward=0.50", log_content)
        self.assertIn("done=False", log_content)

        # Cleanup
        logger.remove(logger_id)

    def test_logging_episode_completion(self):
        """Test that episode completion is logged when logging is enabled."""
        # Setup a string IO as log handler to capture logs
        log_capture = StringIO()
        logger.remove()  # Remove default handlers
        logger_id = logger.add(log_capture, level="DEBUG")

        # Create experience with logging enabled
        exp = Experience(capacity=10, enable_logging=True)

        # Add an episode
        exp.add(Transition(state=1, action=2, reward=0.5, next_state=3, done=False))
        exp.add(Transition(state=3, action=1, reward=1.0, next_state=5, done=False))
        exp.add(Transition(state=5, action=0, reward=2.0, next_state=7, done=True))

        # Check log output
        log_content = log_capture.getvalue()
        self.assertIn("Episode completed with 3 transitions", log_content)
        self.assertIn("total return: 3.50", log_content)

        # Cleanup
        logger.remove(logger_id)

    def test_logging_disabled_no_output(self):
        """Test that no logging occurs when logging is disabled."""
        # Setup a string IO as log handler to capture logs
        log_capture = StringIO()
        logger.remove()  # Remove default handlers
        logger_id = logger.add(log_capture, level="DEBUG")

        # Create experience with logging disabled
        exp = Experience(capacity=10, enable_logging=False)

        # Add transitions
        exp.add(Transition(state=1, action=2, reward=0.5, next_state=3, done=False))
        exp.add(Transition(state=3, action=1, reward=1.0, next_state=5, done=True))

        # Check log output - should be empty
        log_content = log_capture.getvalue()
        self.assertEqual(log_content, "")

        # Cleanup
        logger.remove(logger_id)


if __name__ == "__main__":
    unittest.main()
