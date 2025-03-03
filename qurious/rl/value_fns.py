import pickle
from abc import ABC, abstractmethod

import numpy as np


class ValueFunction(ABC):
    """
    Abstract base class for all value functions.

    A value function estimates the expected return for states or state-action pairs.
    """

    @abstractmethod
    def estimate(self, state, *args):
        """
        Estimate the value for a state or state-action pair.

        Args:
            state: The state
            *args: Additional arguments (e.g., action for ActionValueFunction)

        Returns:
            float: Estimated value
        """
        pass

    @abstractmethod
    def update(self, state, target, *args):
        """
        Update the value function based on a target value.

        Args:
            state: The state
            target (float): Target value
            *args: Additional arguments (e.g., action, alpha for ActionValueFunction)
        """
        pass

    def save(self, filepath):
        """
        Save the value function to a file.

        Args:
            filepath (str): Path to save the value function
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        """
        Load a value function from a file.

        Args:
            filepath (str): Path to the saved value function

        Returns:
            ValueFunction: The loaded value function
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def reset(self):
        """Reset the value function to its initial state."""
        pass


class StateValueFunction(ValueFunction):
    """
    Abstract base class for state value functions (V-functions).

    A state value function estimates the expected return for a state under a policy.
    """

    @abstractmethod
    def estimate(self, state):
        """
        Estimate the value for a state.

        Args:
            state: The state

        Returns:
            float: Estimated value
        """
        pass

    @abstractmethod
    def update(self, state, target, alpha=None):
        """
        Update the value function based on a target value.

        Args:
            state: The state
            target (float): Target value
            alpha (float, optional): Learning rate
        """
        pass


class ActionValueFunction(ValueFunction):
    """
    Abstract base class for action value functions (Q-functions).

    An action value function estimates the expected return for a state-action pair.
    """

    @abstractmethod
    def estimate(self, state, action):
        """
        Estimate the value for a state-action pair.

        Args:
            state: The state
            action: The action

        Returns:
            float: Estimated value
        """
        pass

    @abstractmethod
    def update(self, state, action, target, alpha=None):
        """
        Update the value function based on a target value.

        Args:
            state: The state
            action: The action
            target (float): Target value
            alpha (float, optional): Learning rate
        """
        pass

    @abstractmethod
    def estimate_all_actions(self, state):
        """
        Estimate values for all actions in a state.

        Args:
            state: The state

        Returns:
            numpy.ndarray: Estimated values for all actions
        """
        pass

    def get_best_action(self, state):
        """
        Get the action with highest value for a state.

        Args:
            state: The state

        Returns:
            Action with highest value
        """
        action_values = self.estimate_all_actions(state)
        return np.argmax(action_values)

    def get_best_value(self, state):
        """
        Get the maximum action value for a state.

        Args:
            state: The state

        Returns:
            float: Maximum action value
        """
        action_values = self.estimate_all_actions(state)
        return np.max(action_values)


class TabularStateValueFunction(StateValueFunction):
    """
    A tabular state value function.

    This implements a V-function as a lookup table.
    """

    def __init__(self, n_states, initial_value=0.0):
        """
        Initialize a tabular state value function.

        Args:
            n_states (int): Number of states
            initial_value (float, optional): Initial value for all states
        """
        self.n_states = n_states
        self.initial_value = initial_value
        self.values = np.full(n_states, initial_value)

    def estimate(self, state):
        """
        Estimate the value for a state.

        Args:
            state (int): State index

        Returns:
            float: Estimated value
        """
        return self.values[state]

    def update(self, state, target, alpha=0.1):
        """
        Update the value function based on a target value.

        Args:
            state (int): State index
            target (float): Target value
            alpha (float, optional): Learning rate
        """
        if alpha is None:
            alpha = 0.1

        # V(s) = V(s) + alpha * (target - V(s))
        self.values[state] += alpha * (target - self.values[state])

    def reset(self):
        """Reset the value function to its initial state."""
        self.values.fill(self.initial_value)


class TabularActionValueFunction(ActionValueFunction):
    """
    A tabular action value function.

    This implements a Q-function as a lookup table.
    """

    def __init__(self, n_states, n_actions, initial_value=0.0):
        """
        Initialize a tabular action value function.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
            initial_value (float, optional): Initial value for all state-action pairs
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.initial_value = initial_value
        self.values = np.full((n_states, n_actions), initial_value)

    def estimate(self, state, action):
        """
        Estimate the value for a state-action pair.

        Args:
            state (int): State index
            action (int): Action index

        Returns:
            float: Estimated value
        """
        return self.values[state, action]

    def update(self, state, action, target, alpha=0.1):
        """
        Update the value function based on a target value.

        Args:
            state (int): State index
            action (int): Action index
            target (float): Target value
            alpha (float, optional): Learning rate
        """
        if alpha is None:
            alpha = 0.1

        # Q(s,a) = Q(s,a) + alpha * (target - Q(s,a))
        self.values[state, action] += alpha * (target - self.values[state, action])

    def estimate_all_actions(self, state):
        """
        Estimate values for all actions in a state.

        Args:
            state (int): State index

        Returns:
            numpy.ndarray: Estimated values for all actions
        """
        return self.values[state]

    def reset(self):
        """Reset the value function to its initial state."""
        self.values.fill(self.initial_value)
