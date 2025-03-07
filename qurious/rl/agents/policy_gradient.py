from collections import defaultdict
from typing import Any, Dict, List

import torch

from qurious.rl.agents import Agent
from qurious.rl.policies.llm_policy import TrainableLLMPolicy


class PolicyGradientAgent(Agent):
    """
    Base class for policy gradient reinforcement learning agents.

    Policy gradient methods learn by directly optimizing the policy through
    gradient ascent on the expected return.
    """

    def __init__(self, policy: TrainableLLMPolicy, gamma: float = 0.99, lr: float = 1e-5):
        """
        Initialize a policy gradient agent.

        Args:
            policy: The agent's policy (must be a trainable policy)
            gamma: Discount factor for future rewards, between 0 and 1
            lr: Learning rate for policy optimization
        """
        super().__init__()
        self.policy = policy
        self.gamma = gamma

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(self.policy.model.parameters(), lr=lr)

        # Track metrics
        self.metrics = defaultdict(list)

    def choose_action(self, state: Any) -> Any:
        """
        Select an action using the agent's policy.

        Args:
            state: The current state

        Returns:
            The chosen action
        """
        return self.policy.get_action(state)

    def learn(self, experience: Any) -> Dict[str, float]:
        """
        Update the policy based on experience.

        This generic implementation does nothing and should be overridden.

        Args:
            experience: Experience data (can be a Transition, batch, or episode)

        Returns:
            Dict of metrics from the update
        """
        raise NotImplementedError("Subclasses must implement the learn method")

    def _compute_returns(self, rewards: List[float], dones: List[bool]) -> List[float]:
        """
        Compute discounted returns for a sequence of rewards.

        Args:
            rewards: List of rewards
            dones: List of done flags

        Returns:
            List of discounted returns
        """
        returns = []
        G = 0

        # Iterate backwards through rewards
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = 0  # Reset return for new episodes
            G = r + self.gamma * G
            returns.insert(0, G)  # Insert at front to maintain chronological order

        return returns

    def reset(self):
        """Reset the agent's internal state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()
