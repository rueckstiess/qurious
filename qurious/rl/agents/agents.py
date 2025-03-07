from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Self

from ..environments import Environment
from ..experience import Experience, Transition
from ..policies import DeterministicTabularPolicy, EpsilonGreedyPolicy
from ..value_fns import TabularActionValueFunction


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


class TabularAgent(Agent):
    """
    Base class for agents with tabular representations.

    This agent maintains a policy and optionally state and/or action value functions.
    """

    def __init__(
        self,
        policy,
        value_function=None,
        enable_logging: bool = False,
    ):
        """
        Initialize a tabular agent.

        Args:
            policy (Policy): The agent's policy
            value_function (ValueFunction, optional): The agent's value function
            enable_logging (bool): Whether to log transitions when added
        """
        # Tabular agents always track experience
        super().__init__(track_experience=True, enable_logging=enable_logging)
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

    def learn(self, experience: Any):
        """
        Base implementation of learning method.

        This base implementation does nothing and should be overridden by subclasses.

        Args:
            experience: Experience data from interacting with the environment
        """
        pass

    def reset(self, clear_experience: bool = False):
        """Reset agent's internal state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

        if self.value_function is not None and hasattr(self.value_function, "reset"):
            self.value_function.reset()

        if clear_experience and self.experience is not None:
            self.experience.clear()


class ValueBasedAgent(TabularAgent):
    """
    Agent that uses value functions to make decisions.

    This includes agents implementing Q-learning, SARSA, Expected SARSA, etc.
    """

    def __init__(
        self,
        policy,
        action_value_function,
        gamma=0.99,
        enable_logging: bool = False,
    ):
        """
        Initialize a value-based agent.

        Args:
            policy (Policy): The agent's policy (typically epsilon-greedy or derived from Q)
            action_value_function (ActionValueFunction): The agent's Q-function
            gamma (float, optional): Discount factor for future rewards
            enable_logging (bool): Whether to log transitions when added
        """

        super().__init__(policy, action_value_function, enable_logging=enable_logging)
        self.Q = action_value_function  # Alias for clarity
        self.gamma = gamma

    @classmethod
    def from_env(cls, env: Environment, **kwargs) -> Self:
        """
        Create a default ValueBasedAgent with a simple epsilon-greedy policy and Q-function.

        Args:
            env (Environment): The environment to create the agent for
            **kwargs: Additional parameters for the agent
                - epsilon (float): Initial exploration rate for epsilon-greedy policy
                - decay_rate (float): Rate at which to decay epsilon
                - gamma (float): Discount factor for future rewards for the Q-function
                - enable_logging (bool): Whether to log transitions when added

        Returns:
            ValueBasedAgent: A new agent instance with default components
        """

        # Q-function
        q_function = TabularActionValueFunction(env.num_states, env.num_actions)

        # Base policy (will be updated based on Q-values)
        base_policy = DeterministicTabularPolicy(env.num_states, env.num_actions)

        # Epsilon-greedy exploration policy
        epsilon = kwargs.pop("epsilon", 0.5)
        decay_rate = kwargs.pop("decay_rate", 0.99)
        policy = EpsilonGreedyPolicy(base_policy, epsilon=epsilon, decay_rate=decay_rate)

        # return agent of the class
        gamma = kwargs.pop("gamma", 0.99)
        enable_logging = kwargs.pop("enable_logging", False)
        return cls(policy, q_function, gamma=gamma, enable_logging=enable_logging)

    def _get_transition_tuple(self, transition: Optional[Transition | Tuple] = None) -> Tuple:
        """
        Convert a Transition object to a tuple if necessary. If no transition is provided,
        it retrieves the last transition from the experience buffer.

        Args:
            transition (Transition | tuple): Transition object or tuple

        Returns:
            Tuple: Transition as a tuple
        """
        # if no transition provided, use the last transition from experience
        if transition is None:
            if self.experience is None:
                raise ValueError("No experience buffer available for learning and no transition provided.")
            if len(self.experience) == 0:
                raise ValueError("Experience buffer is empty.")
            transition = self.experience.get_current_transition()

        if isinstance(transition, Transition):
            transition = transition.as_tuple()

        return transition

    def learn(self, transition: Optional[Transition | Tuple] = None):
        """
        Update the agent's value function and policy based on a transition.
        If no transition is provided, it will use the last stored transition
        in the experience buffer.

        For value-based methods, this implements the core TD update.
        Specific algorithms like Q-learning or SARSA should override this.

        Args:
            transition (Transition | tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = self._get_transition_tuple(transition)

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
        self.policy.update_from_value_fn(self.Q)


class QLearningAgent(ValueBasedAgent):
    """
    Agent that implements Q-learning, an off-policy TD control algorithm.

    Q-learning uses the maximum Q-value for the next state regardless of the policy.
    """

    def learn(self, transition: Optional[Transition | Tuple] = None):
        """
        Implement Q-learning update.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = self._get_transition_tuple(transition)

        if done:
            target = reward
        else:
            # Q-learning uses max Q-value for next state (off-policy)
            target = reward + self.gamma * self.Q.get_best_value(next_state)

        # Update the action-value function
        self.Q.update(state, action, target)

        # Update the policy if it depends on the value function
        self.policy.update_from_value_fn(self.Q)


class SarsaAgent(ValueBasedAgent):
    """
    Agent that implements SARSA, an on-policy TD control algorithm.

    SARSA uses the actual next action from the policy.
    """

    def learn(self, transition: Optional[Transition | Tuple] = None):
        """
        Implement SARSA update.

        Args:
            transition (Transition | tuple): (state, action, reward, next_state, done)
        """
        # SARSA exactly matches the default implementation in ValueBasedAgent
        super().learn(transition)


class ExpectedSarsaAgent(ValueBasedAgent):
    """
    Agent that implements Expected SARSA, a compromise between on and off-policy TD.

    Expected SARSA uses the expected value of the next state under the current policy.
    """

    def learn(self, transition: Optional[Transition | Tuple] = None):
        """
        Implement Expected SARSA update.

        Args:
            transition (tuple): (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = self._get_transition_tuple(transition)

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
        self.policy.update_from_value_fn(self.Q)
