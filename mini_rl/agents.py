from abc import ABC, abstractmethod


class Agent(ABC):
    """
    Abstract base class for all reinforcement learning agents.

    An agent interacts with an environment by choosing actions and learning from experience.
    """

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
    def learn(self, experience):
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


class TabularAgent(Agent):
    """
    Base class for agents with tabular representations.

    This agent maintains a policy and optionally state and/or action value functions.
    """

    def __init__(self, policy, value_function=None):
        """
        Initialize a tabular agent.

        Args:
            policy (Policy): The agent's policy
            value_function (ValueFunction, optional): The agent's value function
        """
        self.policy = policy
        self.value_function = value_function

    def choose_action(self, state):
        """
        Select an action using the agent's policy.

        Args:
            state: The current state

        Returns:
            The chosen action
        """
        return self.policy.get_action(state)

    def learn(self, experience):
        """
        Base implementation of learning method.

        This base implementation does nothing and should be overridden by subclasses.

        Args:
            experience: Experience data from interacting with the environment
        """
        pass

    def reset(self):
        """Reset agent's internal state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        if self.value_function is not None and hasattr(self.value_function, "reset"):
            self.value_function.reset()


class ValueBasedAgent(TabularAgent):
    """
    Agent that uses value functions to make decisions.

    This includes agents implementing Q-learning, SARSA, Expected SARSA, etc.
    """

    def __init__(self, policy, action_value_function, gamma=0.99):
        """
        Initialize a value-based agent.

        Args:
            policy (Policy): The agent's policy (typically epsilon-greedy or derived from Q)
            action_value_function (ActionValueFunction): The agent's Q-function
            gamma (float, optional): Discount factor for future rewards
        """
        super().__init__(policy, action_value_function)
        self.Q = action_value_function  # Alias for clarity
        self.gamma = gamma

    def learn(self, transition):
        """
        Update the agent's value function and policy based on a transition.

        For value-based methods, this implements the core TD update.
        Specific algorithms like Q-learning or SARSA should override this.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition

        # This is a default TD update - specific algorithms will override this
        if done:
            target = reward
        else:
            # By default, this is SARSA (on-policy TD)
            next_action = self.choose_action(next_state)
            target = reward + self.gamma * self.Q.estimate(next_state, next_action)

        # Update the action-value function
        self.Q.update(state, action, target)

        # Update the policy if it depends on the value function
        self._update_policy(state)

    def _update_policy(self, state):
        """
        Update the policy based on the current value function.

        Args:
            state: The state to update the policy for
        """
        # Default implementation does nothing
        # Subclasses can override this to update the policy
        pass


class QLearningAgent(ValueBasedAgent):
    """
    Agent that implements Q-learning, an off-policy TD control algorithm.

    Q-learning uses the maximum Q-value for the next state regardless of the policy.
    """

    def learn(self, transition):
        """
        Implement Q-learning update.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition

        if done:
            target = reward
        else:
            # Q-learning uses max Q-value for next state (off-policy)
            target = reward + self.gamma * self.Q.get_best_value(next_state)

        # Update the action-value function
        self.Q.update(state, action, target)


class SarsaAgent(ValueBasedAgent):
    """
    Agent that implements SARSA, an on-policy TD control algorithm.

    SARSA uses the actual next action from the policy.
    """

    def learn(self, transition):
        """
        Implement SARSA update.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        # SARSA exactly matches the default implementation in ValueBasedAgent
        super().learn(transition)


class ExpectedSarsaAgent(ValueBasedAgent):
    """
    Agent that implements Expected SARSA, a compromise between on and off-policy TD.

    Expected SARSA uses the expected value of the next state under the current policy.
    """

    def learn(self, transition):
        """
        Implement Expected SARSA update.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition

        if done:
            target = reward
        else:
            # Calculate expected value based on policy probabilities
            next_action_probs = self.policy.get_action_probabilities(next_state)
            next_action_values = self.Q.estimate_all_actions(next_state)
            expected_value = sum(prob * val for prob, val in zip(next_action_probs, next_action_values))

            target = reward + self.gamma * expected_value

        # Update the action-value function
        self.Q.update(state, action, target)

        # Update the policy if it depends on the value function
        self._update_policy(state)
