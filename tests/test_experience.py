import unittest
import numpy as np
from mini_rl.experience import Experience, Transition


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

    def test_get_current_episode(self):
        """Test current episode access."""
        self.single_episode_exp.add(Transition(0, 1, 0.5, 1, False))  # Start new episode

        current = self.single_episode_exp.get_current_episode()
        self.assertEqual(len(current), 1)
        self.assertFalse(current[0].done)

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


if __name__ == "__main__":
    unittest.main()
