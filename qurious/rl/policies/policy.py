import pickle
from abc import ABC, abstractmethod


class Policy(ABC):
    """
    Abstract base class for all policies.

    A policy defines the behavior of an agent in a Markov Decision Process,
    mapping states to actions or probability distributions over actions.
    """

    @abstractmethod
    def get_action(self, state):
        """
        Get an action for the given state.

        Args:
            state: The current state

        Returns:
            An action to take
        """
        pass

    @abstractmethod
    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state: The current state

        Returns:
            numpy.ndarray: Probability distribution over actions
        """
        pass

    @abstractmethod
    def update(self, state, action, value):
        """
        Update the policy for a state-action pair based on a value.

        Args:
            state: The state
            action: The action
            value: The value used to update the policy
        """
        pass

    @abstractmethod
    def update_from_value_fn(self, value_function):
        """
        Update the policy based on a value function.

        Args:
            value_function: The value function to use for the update
        """
        pass

    def save(self, filepath):
        """
        Save the policy to a file.

        Args:
            filepath (str): Path to save the policy
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a policy from a file.

        Args:
            filepath (str): Path to the saved policy

        Returns:
            Policy: The loaded policy
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)


class DiscretePolicy(Policy):
    """
    Abstract base class for policies with discrete action spaces.
    """

    def __init__(self, n_actions: int):
        """
        Initialize the discrete policy.

        Args:
            n_actions (int): Number of possible actions.
        """
        super().__init__()
        self.n_actions = n_actions
