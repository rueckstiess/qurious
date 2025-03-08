from abc import ABC, abstractmethod
from typing import Any, Optional

from ..experience import Experience, Transition


class Agent(ABC):
    """
    Abstract base class for all reinforcement learning agents.

    An agent interacts with an environment by choosing actions and learning from experience.
    """

    def __init__(self, track_experience: bool = True, enable_logging: bool = False, capacity: Optional[int] = None):
        """Initialize an agent."""
        if track_experience:
            self.track_experience(True, enable_logging, capacity)
        else:
            self.track_experience(False)

    @abstractmethod
    def choose_action(self, state):
        """
        Select an action for the given state.

        Args:
            state: The current state of the environment

        Returns:
            The chosen action
        """
        pass

    @abstractmethod
    def learn(self, experience: Any):
        """
        Update the agent's policy and/or value function based on experience.

        Args:
            experience: Experience data from interacting with the environment
                       (could be a single transition, episode, or batch)
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the agent's internal state (e.g., for a new episode)."""
        pass

    def track_experience(self, enabled: bool, enable_logging: bool = False, capacity: Optional[int] = None) -> None:
        """
        Enable or disable experience tracking.

        Args:
            enabled (bool): Whether to track experience
            enable_logging (bool): Whether to log transitions when added
            capacity (int, optional): Maximum number of transitions to store
        """
        if enabled:
            self.experience = Experience(enable_logging=enable_logging, capacity=capacity)
        else:
            self.experience = None

    def store_experience(self, state, action, reward, next_state, done) -> None:
        """
        Store a transition in the experience buffer if tracking is enabled.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
        """
        if self.experience is not None:
            transition = Transition(state, action, reward, next_state, done)
            self.experience.add(transition)
