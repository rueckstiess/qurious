import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class Transition:
    """A single transition experience."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool


class Experience:
    """Stores and manages agent experience data."""

    def __init__(self, capacity: Optional[int] = None):
        """
        Initialize experience storage.

        Args:
            capacity: Maximum number of transitions to store (None for unlimited)
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_boundaries = []  # Store indices where episodes end
        self._current_episode: List[Transition] = []

    def add(self, transition: Transition) -> None:
        """
        Add a transition to the experience.

        Args:
            transition: The transition to add
        """
        self.buffer.append(transition)
        self._current_episode.append(transition)

        if transition.done:
            self.episode_boundaries.append(len(self.buffer) - 1)
            self._current_episode = []

    def sample_batch(self, batch_size: int) -> List[Transition]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            List of sampled transitions
        """
        indices = np.random.randint(0, len(self.buffer), size=batch_size)
        return [self.buffer[i] for i in indices]

    def sample_episode(self) -> List[Transition]:
        """
        Sample a complete episode randomly.

        Returns:
            List of transitions forming an episode
        """
        if not self.episode_boundaries:
            return []

        # Select random episode end index
        episode_idx = np.random.randint(0, len(self.episode_boundaries))
        end_idx = self.episode_boundaries[episode_idx]
        start_idx = self.episode_boundaries[episode_idx - 1] + 1 if episode_idx > 0 else 0

        return list(self.buffer)[start_idx : end_idx + 1]

    def get_current_episode(self) -> List[Transition]:
        """
        Get transitions from the current ongoing episode.

        Returns:
            List of transitions in the current episode
        """
        return self._current_episode

    def clear(self) -> None:
        """Clear all stored experience."""
        self.buffer.clear()
        self.episode_boundaries.clear()
        self._current_episode.clear()

    @property
    def size(self) -> int:
        """Get the number of stored transitions."""
        return len(self.buffer)
