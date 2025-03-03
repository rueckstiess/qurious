import numpy as np


class MarkovRewardProcess:
    """
    A class representing a Markov Reward Process.

    A Markov Reward Process is a tuple (S, P, R, γ) where:
    - S is a finite set of states
    - P is a state transition probability matrix
    - R is a reward function
    - γ is a discount factor

    Attributes:
        states (int): Number of states in the MRP
        transition_probs (numpy.ndarray): Transition probability matrix with shape (states, states)
                                         P(s' | s) = transition_probs[s, s']
        rewards (numpy.ndarray): Reward matrix with shape (states, states)
                               R(s, s') = rewards[s, s']
        gamma (float): Discount factor, between 0 and 1
        terminal_states (list): List of terminal state indices
    """

    def __init__(self, states, transition_probs=None, rewards=None, gamma=0.99, terminal_states=None):
        """
        Initialize a Markov Reward Process.

        Args:
            states (int): Number of states
            transition_probs (numpy.ndarray, optional): Transition probability matrix
            rewards (numpy.ndarray, optional): Reward matrix
            gamma (float, optional): Discount factor, defaults to 0.99
            terminal_states (list, optional): List of terminal state indices
        """
        self.states = states
        self.gamma = gamma
        self.terminal_states = terminal_states if terminal_states is not None else []

        # Initialize transition probabilities if not provided
        if transition_probs is None:
            # P(s' | s) has shape (states, states)
            self.transition_probs = np.zeros((states, states))
        else:
            self.validate_transition_probs(transition_probs)
            self.transition_probs = transition_probs

        # Initialize rewards if not provided
        if rewards is None:
            # R(s, s') has shape (states, states)
            self.rewards = np.zeros((states, states))
        else:
            self.validate_rewards(rewards)
            self.rewards = rewards

        # Derived data
        self._calculate_expected_rewards()

    def validate_transition_probs(self, transition_probs):
        """
        Validate the transition probability matrix.

        Args:
            transition_probs (numpy.ndarray): Transition probability matrix to validate

        Raises:
            ValueError: If the matrix shape is incorrect or probabilities don't sum to 1
        """
        expected_shape = (self.states, self.states)
        if transition_probs.shape != expected_shape:
            raise ValueError(
                f"Transition probability matrix has shape {transition_probs.shape}, expected {expected_shape}"
            )

        # Check if probabilities sum to 1 for each state
        for s in range(self.states):
            prob_sum = np.sum(transition_probs[s])
            if not np.isclose(prob_sum, 1.0) and prob_sum > 0:
                raise ValueError(f"Transition probabilities for state {s} sum to {prob_sum}, expected 1.0")

    def validate_rewards(self, rewards):
        """
        Validate the reward matrix.

        Args:
            rewards (numpy.ndarray): Reward matrix to validate

        Raises:
            ValueError: If the matrix shape is incorrect
        """
        expected_shape = (self.states, self.states)
        if rewards.shape != expected_shape:
            raise ValueError(f"Reward matrix has shape {rewards.shape}, expected {expected_shape}")

    def _calculate_expected_rewards(self):
        """Calculate the expected rewards for each state."""
        # R(s) = ∑s' P(s' | s) * R(s, s')
        self.expected_rewards = np.sum(self.transition_probs * self.rewards, axis=1)

    def get_transition_prob(self, state, next_state):
        """
        Get the transition probability P(next_state | state).

        Args:
            state (int): Current state index
            next_state (int): Next state index

        Returns:
            float: Transition probability
        """
        return self.transition_probs[state, next_state]

    def get_reward(self, state, next_state):
        """
        Get the reward R(state, next_state).

        Args:
            state (int): Current state index
            next_state (int): Next state index

        Returns:
            float: Reward value
        """
        return self.rewards[state, next_state]

    def get_expected_reward(self, state):
        """
        Get the expected reward for a state.

        Args:
            state (int): State index

        Returns:
            float: Expected reward
        """
        return self.expected_rewards[state]

    def is_terminal(self, state):
        """
        Check if a state is terminal.

        Args:
            state (int): State index

        Returns:
            bool: True if the state is terminal, False otherwise
        """
        return state in self.terminal_states

    def set_transition_prob(self, state, next_state, prob):
        """
        Set the transition probability P(next_state | state).

        Args:
            state (int): Current state index
            next_state (int): Next state index
            prob (float): Transition probability value
        """
        self.transition_probs[state, next_state] = prob
        self._calculate_expected_rewards()  # Update expected rewards

    def set_reward(self, state, next_state, reward):
        """
        Set the reward R(state, next_state).

        Args:
            state (int): Current state index
            next_state (int): Next state index
            reward (float): Reward value
        """
        self.rewards[state, next_state] = reward
        self._calculate_expected_rewards()  # Update expected rewards

    def get_next_state_distribution(self, state):
        """
        Get the distribution over next states given a state.

        Args:
            state (int): Current state index

        Returns:
            numpy.ndarray: Probability distribution over next states
        """
        return self.transition_probs[state]

    def sample_next_state(self, state):
        """
        Sample a next state from the transition distribution.

        Args:
            state (int): Current state index

        Returns:
            int: Sampled next state index
        """
        next_state_probs = self.transition_probs[state]
        return np.random.choice(self.states, p=next_state_probs)

    def get_possible_next_states(self, state):
        """
        Get all possible next states with non-zero transition probability.

        Args:
            state (int): Current state index

        Returns:
            list: List of possible next state indices
        """
        return np.where(self.transition_probs[state] > 0)[0].tolist()

    def calculate_state_value_function(self, theta=1e-6, max_iterations=1000):
        """
        Calculate the state value function using value iteration.

        Args:
            theta (float, optional): Convergence threshold
            max_iterations (int, optional): Maximum number of iterations

        Returns:
            numpy.ndarray: State value function array
        """
        V = np.zeros(self.states)

        for i in range(max_iterations):
            delta = 0
            for s in range(self.states):
                if self.is_terminal(s):
                    V[s] = 0
                    continue

                v = V[s]

                # Bellman equation for MRP: V(s) = R(s) + γ * ∑s' P(s'|s) * V(s')
                expected_next_value = np.sum(self.transition_probs[s] * V)
                V[s] = self.expected_rewards[s] + self.gamma * expected_next_value

                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        return V

    @classmethod
    def from_mdp_and_policy(cls, mdp, policy):
        """
        Create an MRP from an MDP and a policy.

        Args:
            mdp (MarkovDecisionProcess): The MDP
            policy (numpy.ndarray): Policy matrix with shape (states, actions)
                                   policy[s, a] = probability of taking action a in state s

        Returns:
            MarkovRewardProcess: The induced MRP
        """
        states = mdp.states

        # Initialize transition probabilities and rewards for the MRP
        transition_probs = np.zeros((states, states))
        rewards = np.zeros((states, states))

        # For each state and next state
        for s in range(states):
            for s_next in range(states):
                # Sum over all actions
                for a in range(mdp.actions):
                    # P(s'|s) = ∑a π(a|s) * P(s'|s,a)
                    transition_prob = policy[s, a] * mdp.transition_probs[s, a, s_next]
                    transition_probs[s, s_next] += transition_prob

                    # R(s,s') = ∑a π(a|s) * P(s'|s,a) * R(s,a,s') / P(s'|s)
                    if transition_prob > 0:
                        rewards[s, s_next] += transition_prob * mdp.rewards[s, a, s_next]

        # Normalize rewards by transition probabilities
        for s in range(states):
            for s_next in range(states):
                if transition_probs[s, s_next] > 0:
                    rewards[s, s_next] /= transition_probs[s, s_next]

        return cls(states, transition_probs, rewards, mdp.gamma, mdp.terminal_states)
