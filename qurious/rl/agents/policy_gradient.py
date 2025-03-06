from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from qurious.rl.agents import Agent
from qurious.rl.experience import Transition
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


class REINFORCEAgent(PolicyGradientAgent):
    """
    Agent implementing the REINFORCE algorithm (vanilla policy gradient).

    REINFORCE learns by sampling trajectories and updating the policy to
    increase the probability of actions that led to higher returns.
    """

    def learn(self, experience: Union[List[Transition], Tuple]) -> Dict[str, float]:
        """
        Update the policy using the REINFORCE algorithm.

        If a single transition is provided, it will be stored until an episode is complete.
        If an episode is complete, the policy will be updated.

        Args:
            experience: An episode of transitions or single transition

        Returns:
            Dict of metrics from the update
        """
        # Handle different input types
        if isinstance(experience, Transition):
            # If a single transition, store it and only update when episode is complete
            if not experience.done:
                return {}  # No update until episode is complete
            # If episode is complete, use stored transitions
            if self.experience is None:
                raise ValueError("Experience tracking must be enabled for REINFORCEAgent")
            episode = self.experience.get_current_episode()
        elif isinstance(experience, list) and all(isinstance(t, Transition) for t in experience):
            # If a list of transitions, update immediately
            episode = experience
        else:
            raise ValueError("Experience must be a Transition or list of Transitions")

        # Extract states, actions, rewards from the episode
        states = [t.state for t in episode]
        actions = [t.action for t in episode]
        rewards = [t.reward for t in episode]
        dones = [t.done for t in episode]

        # Compute returns (discounted rewards)
        returns = self._compute_returns(rewards, dones)

        # Normalize returns for stability
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Update policy
        self.policy.train()
        loss = self._compute_policy_loss(states, actions, returns)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Track metrics
        loss_value = loss.item()
        self.metrics["loss"].append(loss_value)
        self.metrics["episode_return"].append(sum(rewards))
        self.metrics["episode_length"].append(len(rewards))

        return {"loss": loss_value, "episode_return": sum(rewards), "episode_length": len(rewards)}

    def _compute_policy_loss(self, states, actions, returns):
        """
        Compute the REINFORCE loss: -log(π(a|s)) * G

        Args:
            states: List of states
            actions: List of actions taken
            returns: List of discounted returns

        Returns:
            torch.Tensor: Loss value
        """
        loss = 0
        for state, action, G in zip(states, actions, returns):
            # Get log probability of action
            log_prob = self.policy._get_action_logprobs(state, action)

            # Convert return to tensor if it's not already
            if not isinstance(G, torch.Tensor):
                G = torch.tensor(G, device=log_prob.device)

            # Policy gradient loss: -log(π(a|s)) * G
            step_loss = -log_prob * G
            loss += step_loss

        # Average loss over trajectory
        return loss / len(states)


class PPOAgent(PolicyGradientAgent):
    """
    Agent implementing Proximal Policy Optimization (PPO).

    PPO learns by sampling trajectories and updating the policy with a
    clipped objective to prevent large policy changes.
    """

    def __init__(
        self,
        policy: TrainableLLMPolicy,
        value_model=None,  # Can be separate or shared with policy
        gamma: float = 0.99,
        lr: float = 1e-5,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        gae_lambda: float = 0.95,
        epochs_per_update: int = 4,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize a PPO agent.

        Args:
            policy: The agent's policy (must be a trainable policy)
            value_model: Model for value function approximation (defaults to a separate head on the policy)
            gamma: Discount factor for future rewards
            lr: Learning rate for optimization
            clip_ratio: PPO clipping parameter (epsilon)
            value_coef: Coefficient for value function loss
            entropy_coef: Coefficient for entropy bonus
            gae_lambda: GAE lambda parameter
            epochs_per_update: Number of optimization epochs per update
            batch_size: Batch size for updates (None = full batch)
        """
        super().__init__(policy, gamma, lr)
        self.value_model = value_model
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size

        # If no separate value model is provided, initialize a value head
        if self.value_model is None:
            # Here you would typically add a value head to the policy model
            # For this example, we'll define a placeholder implementation
            raise NotImplementedError("A value model is required. Automatic value head creation not yet implemented.")

    def learn(self, experience: Union[List[Transition], Tuple]) -> Dict[str, float]:
        """
        Update the policy using the PPO algorithm.

        Args:
            experience: An episode of transitions or batch of transitions

        Returns:
            Dict of metrics from the update
        """
        # This is a placeholder implementation
        # A full PPO implementation would:
        # 1. Collect trajectories
        # 2. Compute advantages using GAE
        # 3. Cache old policy probabilities
        # 4. Perform multiple epochs of updates with clipping

        raise NotImplementedError("PPO implementation is not complete")
