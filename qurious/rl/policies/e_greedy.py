import numpy as np

from qurious.rl.policies.policy import DiscretePolicy


class EpsilonGreedyPolicy(DiscretePolicy):
    """
    An epsilon-greedy policy based on a deterministic policy.

    With probability epsilon, takes a random action.
    With probability 1-epsilon, takes the action from the base policy.
    Epsilon can decay over time if decay_rate is specified.
    """

    def __init__(self, base_policy, epsilon=0.1, decay_rate=None, min_epsilon=0.01):
        """
        Initialize an epsilon-greedy policy.

        Args:
            base_policy (Policy): The base policy to use for greedy actions
            epsilon (float, optional): Initial probability of taking a random action
            decay_rate (float, optional): Rate at which epsilon decays (multiply by this value)
            min_epsilon (float, optional): Minimum value for epsilon after decay
        """
        super().__init__(base_policy.n_actions)
        self.base_policy = base_policy
        self._initial_epsilon = epsilon
        self._epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    @property
    def epsilon(self):
        """Get current epsilon value."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value):
        """
        Set a new epsilon value.

        Args:
            value (float): New epsilon value
        """
        if not (0 <= value <= 1):
            raise ValueError("Epsilon must be between 0 and 1.")
        self._epsilon = value

    def decay_epsilon(self):
        """
        Decay the epsilon value if decay_rate is set.
        Returns:
            float: New epsilon value
        """
        if self.decay_rate is not None:
            self._epsilon = max(self.min_epsilon, self._epsilon * self.decay_rate)
        return self._epsilon

    def reset_epsilon(self):
        """Reset epsilon to its initial value."""
        self._epsilon = self._initial_epsilon

    def get_action(self, state):
        """
        Get an action for the given state using epsilon-greedy strategy.

        Args:
            state (int): State index

        Returns:
            int: Action index
        """
        if np.random.random() < self._epsilon:
            # Random action
            return np.random.choice(self.n_actions)
        else:
            # Greedy action
            return self.base_policy.get_action(state)

    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state (int): State index

        Returns:
            numpy.ndarray: Probability distribution over actions
        """
        base_probs = self.base_policy.get_action_probabilities(state)

        # Epsilon probability of random action, (1-epsilon) for base policy
        probs = np.ones(self.n_actions) * (self._epsilon / self.n_actions)
        probs += (1 - self._epsilon) * base_probs

        return probs

    def update(self, state, action, value):
        """
        Update the base policy.

        Args:
            state (int): State index
            action (int): Action index
            value (float): The value used to update the policy
        """
        self.base_policy.update(state, action, value)

    def update_from_value_fn(self, value_function):
        """
        Update the base policy.

        Args:
            value_function: The value function to use for the update
        """
        self.base_policy.update_from_value_fn(value_function)
