import numpy as np
from abc import ABC, abstractmethod
import pickle


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


class TabularPolicy(Policy):
    """
    Base class for tabular policies.

    Tabular policies explicitly represent the policy for each state in a table.
    """

    def __init__(self, n_states, n_actions):
        """
        Initialize a tabular policy.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
        """
        self.n_states = n_states
        self.n_actions = n_actions


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


class StochasticTabularPolicy(TabularPolicy):
    """
    A stochastic tabular policy.

    For each state, this policy defines a probability distribution over actions.
    """

    def __init__(self, n_states, n_actions, initialization="uniform"):
        """
        Initialize a stochastic tabular policy.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
            initialization (str, optional): How to initialize the policy:
                - 'uniform': Uniform distribution over actions
                - 'random': Random probability distribution over actions
        """
        super().__init__(n_states, n_actions)

        # Initialize policy probabilities
        self.policy = np.zeros((n_states, n_actions))

        if initialization == "uniform":
            self.policy.fill(1.0 / n_actions)
        elif initialization == "random":
            # Generate random probabilities that sum to 1 for each state
            for s in range(n_states):
                self.policy[s] = np.random.dirichlet(np.ones(n_actions))
        else:
            raise ValueError(f"Unknown initialization: {initialization}")

    def get_action(self, state):
        """
        Sample an action from the policy for the given state.

        Args:
            state (int): State index

        Returns:
            int: Sampled action index
        """
        return np.random.choice(self.n_actions, p=self.policy[state])

    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state (int): State index

        Returns:
            numpy.ndarray: Probability distribution over actions
        """
        return self.policy[state]

    def update(self, state, action, value):
        """
        Update the policy for a state-action pair based on a value.

        This implementation uses a soft-max update rule.

        Args:
            state (int): State index
            action (int): Action index
            value (float): The value used to update the policy
        """
        # Initialize temperature parameter for softmax
        temperature = 1.0

        # Get current values for all actions in the state
        values = np.zeros(self.n_actions)
        values[action] = value

        # Calculate softmax probabilities
        exp_values = np.exp(values / temperature)
        self.policy[state] = exp_values / np.sum(exp_values)

    def update_from_value_fn(self, value_function):
        """
        Update the policy based on a value function.

        This implementation uses a soft-max update rule.

        Args:
            value_function: The value function to use for the update
        """
        # Initialize temperature parameter for softmax
        temperature = 1.0

        for state in range(self.n_states):
            # Get current values for all actions in the state
            values = value_function[state]

            # Calculate softmax probabilities
            exp_values = np.exp(values / temperature)
            self.policy[state] = exp_values / np.sum(exp_values)


class EpsilonGreedyPolicy(TabularPolicy):
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
        super().__init__(base_policy.n_states, base_policy.n_actions)
        self.base_policy = base_policy
        self._initial_epsilon = epsilon
        self._epsilon = epsilon
        self.decay_rate = decay_rate
        self.min_epsilon = min_epsilon

    @property
    def epsilon(self):
        """Get current epsilon value."""
        return self._epsilon

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


class SoftmaxPolicy(TabularPolicy):
    """
    A policy that uses a softmax distribution over action values.

    For each state, actions are chosen with probability proportional to e^(Q(s,a)/temperature).
    """

    def __init__(self, n_states, n_actions, temperature=1.0):
        """
        Initialize a softmax policy.

        Args:
            n_states (int): Number of states
            n_actions (int): Number of actions
            temperature (float, optional): Temperature parameter for softmax
        """
        super().__init__(n_states, n_actions)
        self.temperature = temperature
        self.action_values = np.zeros((n_states, n_actions))

    def get_action(self, state):
        """
        Sample an action from the policy for the given state.

        Args:
            state (int): State index

        Returns:
            int: Sampled action index
        """
        probs = self.get_action_probabilities(state)
        return np.random.choice(self.n_actions, p=probs)

    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state (int): State index

        Returns:
            numpy.ndarray: Probability distribution over actions
        """
        # Calculate softmax probabilities
        values = self.action_values[state]
        exp_values = np.exp(values / self.temperature)
        return exp_values / np.sum(exp_values)

    def update(self, state, action, value):
        """
        Update the action value for a state-action pair.

        Args:
            state (int): State index
            action (int): Action index
            value (float): The new action value
        """
        self.action_values[state, action] = value
