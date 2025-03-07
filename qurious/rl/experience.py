import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

import numpy as np

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


@dataclass
class Transition:
    """A single transition experience."""

    state: Any
    action: Any
    reward: float
    next_state: Any
    done: bool

    def as_tuple(self) -> Tuple:
        return (self.state, self.action, self.reward, self.next_state, self.done)


class Experience:
    """Stores and manages agent experience data."""

    def __init__(self, capacity: Optional[int] = None, enable_logging: bool = False):
        """
        Initialize experience storage.

        Args:
            capacity: Maximum number of transitions to store (None for unlimited)
            enable_logging: Whether to log transitions when added (default: False)
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.episode_boundaries = []  # Store indices where episodes end
        self._current_episode: List[Transition] = []
        self._last_completed_episode: List[Transition] = []
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(__name__)

    def add(self, transition: Transition) -> None:
        """
        Add a transition to the experience.

        Args:
            transition: The transition to add
        """
        self.buffer.append(transition)
        self._current_episode.append(transition)

        if self.enable_logging:
            self.logger.info(
                f"Added transition: state={transition.state}, action={transition.action}, "
                f"reward={transition.reward:.4f}, done={transition.done}"
            )

        if transition.done:
            self.episode_boundaries.append(len(self.buffer) - 1)
            self._last_completed_episode = self._current_episode
            self._current_episode = []

            if self.enable_logging:
                self.logger.info(
                    f"Episode completed with {len(self._last_completed_episode)} transitions, "
                    f"total return: {sum(t.reward for t in self._last_completed_episode):.4f}"
                )

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

    def get_current_transition(self) -> Optional[Transition]:
        """
        Get the most recent transition added to the experience.

        Returns:
            The most recent transition, or None if no transitions have been added
        """
        if self.buffer:
            return self.buffer[-1]

        return None

    def get_current_episode(self) -> List[Transition]:
        """
        Get transitions from the current ongoing episode.
        If the current episode is empty and an episode just completed,
        returns the last completed episode instead.

        Returns:
            List of transitions in the current episode or last completed episode
        """
        if self._current_episode:
            return self._current_episode
        return self._last_completed_episode

    def clear(self) -> None:
        """Clear all stored experience."""
        self.buffer.clear()
        self.episode_boundaries.clear()
        self._current_episode.clear()
        self._last_completed_episode.clear()

    @property
    def size(self) -> int:
        """Get the number of stored transitions."""
        return len(self.buffer)

    def __len__(self) -> int:
        """Get the number of stored transitions."""
        return self.size

    def __iter__(self) -> Iterator[Transition]:
        """
        Iterate over all transitions in order of recording.

        Returns:
            Iterator over transitions
        """
        return iter(self.buffer)

    def iter_episodes(self) -> Iterator[List[Transition]]:
        """
        Iterate over complete episodes in order of recording.

        Returns:
            Iterator over episodes (lists of transitions)
        """
        if not self.episode_boundaries:
            return iter([])

        start_idx = 0
        buffer_list = list(self.buffer)

        for end_idx in self.episode_boundaries:
            yield buffer_list[start_idx : end_idx + 1]
            start_idx = end_idx + 1

        # If there's an incomplete episode at the end, yield it too
        if self._current_episode:
            yield self._current_episode
