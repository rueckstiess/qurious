import numpy as np


class MarkovDecisionProcess:
    """
    A class representing a Markov Decision Process.

    Attributes:
        states (int): Number of states in the MDP
        actions (int): Number of actions in the MDP
        transition_probs (numpy.ndarray): Transition probability matrix with shape (states, actions, states)
                                         P(s' | s, a) = transition_probs[s, a, s']
        rewards (numpy.ndarray): Reward matrix with shape (states, actions, states)
                                R(s, a, s') = rewards[s, a, s']
        gamma (float): Discount factor, between 0 and 1
        terminal_states (list): List of terminal state indices
    """

    def __init__(self, states, actions, transition_probs=None, rewards=None, gamma=0.99, terminal_states=None):
        """
        Initialize a Markov Decision Process.

        Args:
            states (int): Number of states
            actions (int): Number of actions
            transition_probs (numpy.ndarray, optional): Transition probability matrix
            rewards (numpy.ndarray, optional): Reward matrix
            gamma (float, optional): Discount factor, defaults to 0.99
            terminal_states (list, optional): List of terminal state indices
        """
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.terminal_states = terminal_states if terminal_states is not None else []

        # Initialize transition probabilities if not provided
        if transition_probs is None:
            # P(s' | s, a) has shape (states, actions, states)
            self.transition_probs = np.zeros((states, actions, states))
        else:
            self.validate_transition_probs(transition_probs)
            self.transition_probs = transition_probs

        # Initialize rewards if not provided
        if rewards is None:
            # R(s, a, s') has shape (states, actions, states)
            self.rewards = np.zeros((states, actions, states))
        else:
            self.validate_rewards(rewards)
            self.rewards = rewards

    def validate_transition_probs(self, transition_probs):
        """
        Validate the transition probability matrix.

        Args:
            transition_probs (numpy.ndarray): Transition probability matrix to validate

        Raises:
            ValueError: If the matrix shape is incorrect or probabilities don't sum to 1
        """
        expected_shape = (self.states, self.actions, self.states)
        if transition_probs.shape != expected_shape:
            raise ValueError(
                f"Transition probability matrix has shape {transition_probs.shape}, expected {expected_shape}"
            )

        # Check if probabilities sum to 1 for each state-action pair
        for s in range(self.states):
            for a in range(self.actions):
                prob_sum = np.sum(transition_probs[s, a])
                if not np.isclose(prob_sum, 1.0) and prob_sum > 0:
                    raise ValueError(
                        f"Transition probabilities for state {s} and action {a} sum to {prob_sum}, expected 1.0"
                    )

    def validate_rewards(self, rewards):
        """
        Validate the reward matrix.

        Args:
            rewards (numpy.ndarray): Reward matrix to validate

        Raises:
            ValueError: If the matrix shape is incorrect
        """
        expected_shape = (self.states, self.actions, self.states)
        if rewards.shape != expected_shape:
            raise ValueError(f"Reward matrix has shape {rewards.shape}, expected {expected_shape}")

    def get_transition_prob(self, state, action, next_state):
        """
        Get the transition probability P(next_state | state, action).

        Args:
            state (int): Current state index
            action (int): Action index
            next_state (int): Next state index

        Returns:
            float: Transition probability
        """
        return self.transition_probs[state, action, next_state]

    def get_reward(self, state, action, next_state):
        """
        Get the reward R(state, action, next_state).

        Args:
            state (int): Current state index
            action (int): Action index
            next_state (int): Next state index

        Returns:
            float: Reward value
        """
        return self.rewards[state, action, next_state]

    def get_expected_reward(self, state, action):
        """
        Get the expected reward for a state-action pair.

        Args:
            state (int): State index
            action (int): Action index

        Returns:
            float: Expected reward
        """
        return np.sum(self.transition_probs[state, action] * self.rewards[state, action])

    def is_terminal(self, state):
        """
        Check if a state is terminal.

        Args:
            state (int): State index

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        return state in self.terminal_states

    def set_transition_prob(self, state, action, next_state, prob):
        """
        Set the transition probability P(next_state | state, action).

        Args:
            state (int): Current state index
            action (int): Action index
            next_state (int): Next state index
            prob (float): Transition probability value
        """
        self.transition_probs[state, action, next_state] = prob

    def set_reward(self, state, action, next_state, reward):
        """
        Set the reward R(state, action, next_state).

        Args:
            state (int): Current state index
            action (int): Action index
            next_state (int): Next state index
            reward (float): Reward value
        """
        self.rewards[state, action, next_state] = reward

    def get_next_state_distribution(self, state, action):
        """
        Get the distribution over next states given a state-action pair.

        Args:
            state (int): Current state index
            action (int): Action index

        Returns:
            numpy.ndarray: Probability distribution over next states
        """
        return self.transition_probs[state, action]

    def sample_next_state(self, state, action):
        """
        Sample a next state from the transition distribution.

        Args:
            state (int): Current state index
            action (int): Action index

        Returns:
            int: Sampled next state index
        """
        next_state_probs = self.transition_probs[state, action]
        return np.random.choice(self.states, p=next_state_probs)

    def get_possible_next_states(self, state, action):
        """
        Get all possible next states with non-zero transition probability.

        Args:
            state (int): Current state index
            action (int): Action index

        Returns:
            list: List of possible next state indices
        """
        return np.where(self.transition_probs[state, action] > 0)[0].tolist()
