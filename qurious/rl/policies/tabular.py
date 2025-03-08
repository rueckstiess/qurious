import numpy as np

from qurious.rl.policies.policy import DiscretePolicy


class TabularPolicy(DiscretePolicy):
    """
    Base class for tabular policies.

    Tabular policies have discerete state and action spaces, and explicitly represent
    the policy for each state in a table.
    """

    def __init__(self, n_states, n_actions):
        """
        Initialize a tabular policy.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
        """
        super().__init__(n_actions)
        self.n_states = n_states


class DeterministicTabularPolicy(TabularPolicy):
    """
    A deterministic tabular policy.

    For each state, this policy maps to a single action with probability 1.
    """

    def __init__(self, n_states, n_actions, default_action=0):
        """
        Initialize a deterministic tabular policy.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
            default_action (int, optional): Default action for all states
        """
        super().__init__(n_states, n_actions)
        self.policy = np.full(n_states, default_action, dtype=int)

    def get_action(self, state):
        """
        Get the action for the given state.

        Args:
            state (int): State index

        Returns:
            int: Action index
        """
        return self.policy[state]

    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state (int): State index

        Returns:
            numpy.ndarray: One-hot probability distribution over actions
        """
        probs = np.zeros(self.n_actions)
        probs[self.policy[state]] = 1.0
        return probs

    def update(self, state, action, value=None):
        """
        Update the policy for a state.

        For deterministic policies, this simply sets the action for the state.

        Args:
            state (int): State index
            action (int): Action index
            value: Ignored for deterministic policies
        """
        self.policy[state] = action

    def update_from_value_fn(self, value_function):
        """
        Update the policy based on a value function.

        Args:
            value_function: The value function to use for the update
        """
        for state in range(self.n_states):
            self.policy[state] = np.argmax(value_function.estimate_all_actions(state))
