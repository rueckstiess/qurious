from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from qurious.rl.agents.policy_gradient import PolicyGradientAgent
from qurious.rl.experience import Transition
from qurious.rl.policies.llm_policy import TrainableLLMPolicy


class GRPOAgent(PolicyGradientAgent):
    """
    Agent implementing Group Relative Policy Optimization (GRPO).

    GRPO is a variant of PPO that:
    1. Eliminates the need for a value function by using group-based advantage estimation
    2. Directly incorporates KL divergence in the loss function
    3. Is designed for efficient single-update training

    Reference: DeepSeekMath paper (https://arxiv.org/abs/2402.03300)
    """

    def __init__(
        self,
        policy: TrainableLLMPolicy,
        gamma: float = 0.99,
        lr: float = 1e-5,
        clip_ratio: float = 0.2,
        beta: float = 0.04,
        group_size: int = 16,
        process_supervision: bool = False,
        reference_policy: Optional[TrainableLLMPolicy] = None,
    ):
        """
        Initialize a GRPO agent.

        Args:
            policy: The agent's policy (must be a trainable policy)
            gamma: Discount factor for future rewards
            lr: Learning rate for policy optimization
            clip_ratio: PPO clipping parameter (epsilon)
            beta: Coefficient for KL regularization
            group_size: Number of outputs to sample for each input for group-based advantage estimation
            process_supervision: Whether to use process (step-by-step) supervision or outcome supervision
            reference_policy: Reference policy for KL divergence (defaults to a copy of the initial policy)
        """
        super().__init__(policy, gamma, lr)
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.group_size = group_size
        self.process_supervision = process_supervision

        # If no reference policy is provided, use the current policy
        if reference_policy is None:
            # In a real implementation, you would create a copy of the policy
            # This is a placeholder - actual implementation would depend on your policy structure
            self.reference_policy = policy
        else:
            self.reference_policy = reference_policy

        # Add a metric to track KL divergence
        self.metrics["kl_divergence"] = []

    def learn(self, experience: Union[List[Transition], Tuple]) -> Dict[str, float]:
        """
        Update the policy using the GRPO algorithm.

        For GRPO, we need multiple outputs for the same input to compute group-relative advantages.
        This implementation assumes that experience contains transitions from multiple outputs
        for the same input, organized into groups.

        Args:
            experience: A list of transitions or groups of transitions

        Returns:
            Dict of metrics from the update
        """
        # If a single transition, we need a larger batch to perform GRPO
        if isinstance(experience, Transition):
            # If experience tracking is not enabled or the current episode doesn't
            # have enough samples, we can't do a GRPO update
            if self.experience is None or len(self.experience.buffer) < self.group_size:
                return {}

            # Use the current batch of experiences
            group_experiences = self._organize_experience_into_groups(self.experience.buffer)
        elif isinstance(experience, list) and all(isinstance(t, Transition) for t in experience):
            # Organize transitions into groups based on input questions
            group_experiences = self._organize_experience_into_groups(experience)
        else:
            raise ValueError("Experience must be a Transition or list of Transitions")

        # Process each group of experiences
        total_loss = 0
        total_kl_div = 0
        num_groups = len(group_experiences)

        for group in group_experiences:
            # Extract states, actions, rewards from the group
            states = [t.state for t in group]
            actions = [t.action for t in group]
            rewards = [t.reward for t in group]
            dones = [t.done for t in group]

            # Calculate group-relative advantages
            advantages = self._compute_group_relative_advantages(group, rewards, dones)

            # Update policy
            self.policy.train()
            loss, kl_div = self._compute_grpo_loss(states, actions, advantages)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_kl_div += kl_div.item()

        # Track metrics
        avg_loss = total_loss / max(num_groups, 1)
        avg_kl_div = total_kl_div / max(num_groups, 1)

        self.metrics["loss"].append(avg_loss)
        self.metrics["kl_divergence"].append(avg_kl_div)

        return {"loss": avg_loss, "kl_divergence": avg_kl_div}

    def _organize_experience_into_groups(self, experiences: List[Transition]) -> List[List[Transition]]:
        """
        Organize experiences into groups based on input questions.

        For GRPO, we need to group transitions by their input states (questions).
        This is a simplified implementation - in practice, you might need a more
        sophisticated grouping mechanism based on your specific application.

        Args:
            experiences: List of transitions

        Returns:
            List of groups, where each group contains transitions with the same input state
        """
        # This is a simplified implementation. In practice, you'd group by question ID or similar
        # For now, we'll just create a single group of size group_size
        if len(experiences) < self.group_size:
            return [experiences]

        # In a real implementation, you would group transitions by their input state
        # For simplicity, we'll just create groups of size group_size
        groups = []
        for i in range(0, len(experiences), self.group_size):
            group = experiences[i : i + self.group_size]
            if len(group) == self.group_size:  # Only use complete groups
                groups.append(group)

        return groups

    def _compute_group_relative_advantages(
        self, group: List[Transition], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """
        Compute group-relative advantages.

        For GRPO, advantages are computed by normalizing rewards relative to the group.

        Args:
            group: Group of transitions
            rewards: List of rewards for each transition
            dones: List of done flags for each transition

        Returns:
            List of advantages
        """
        if self.process_supervision:
            # For process supervision, we compute step-wise rewards
            # This is simplified - in a real implementation, you'd need
            # to extract step-wise rewards from your reward model
            return self._compute_process_supervision_advantages(group, rewards, dones)
        else:
            # For outcome supervision, we use the final rewards only
            return self._compute_outcome_supervision_advantages(rewards)

    def _compute_outcome_supervision_advantages(self, rewards: List[float]) -> List[float]:
        """
        Compute advantages for outcome supervision (end-reward only).

        Args:
            rewards: List of rewards for each transition

        Returns:
            List of normalized advantages
        """
        # Convert rewards to numpy array for easier operations
        rewards_array = np.array(rewards)

        # Normalize rewards relative to the group
        if len(rewards_array) > 1 and rewards_array.std() > 0:
            normalized_rewards = (rewards_array - rewards_array.mean()) / (rewards_array.std() + 1e-8)
        else:
            normalized_rewards = rewards_array

        return normalized_rewards.tolist()

    def _compute_process_supervision_advantages(
        self, group: List[Transition], rewards: List[float], dones: List[bool]
    ) -> List[float]:
        """
        Compute advantages for process supervision (step-wise rewards).

        Args:
            group: Group of transitions
            rewards: List of rewards for each transition
            dones: List of done flags for each transition

        Returns:
            List of advantages
        """
        # For process supervision, we'd need to map rewards to individual tokens/steps
        # This is a simplified implementation - in practice, you would have a more
        # sophisticated mapping based on your reward model

        # Compute discounted returns
        returns = self._compute_returns(rewards, dones)

        # Normalize returns relative to the group
        returns_array = np.array(returns)
        if len(returns_array) > 1 and returns_array.std() > 0:
            normalized_returns = (returns_array - returns_array.mean()) / (returns_array.std() + 1e-8)
        else:
            normalized_returns = returns_array

        return normalized_returns.tolist()

    def _compute_grpo_loss(
        self, states: List[Any], actions: List[Any], advantages: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the GRPO loss: policy gradient loss with clipping + KL regularization.

        Args:
            states: List of states
            actions: List of actions
            advantages: List of advantage estimates

        Returns:
            Tuple of (total loss, KL divergence)
        """
        loss = 0
        kl_div = 0

        for state, action, advantage in zip(states, actions, advantages):
            # Convert advantage to tensor if it's not already
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, device=self.policy.device)

            # Get log probability of action under current policy
            log_prob = self.policy._get_action_logprobs(state, action)

            # Get log probability of action under reference policy
            with torch.no_grad():
                ref_log_prob = self.reference_policy._get_action_logprobs(state, action)
                old_log_prob = self.policy._get_action_logprobs(state, action)

            # Compute importance ratio
            ratio = torch.exp(log_prob - old_log_prob)

            # Compute clipped surrogate objective
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            surrogate1 = ratio * advantage
            surrogate2 = clipped_ratio * advantage
            policy_loss = -torch.min(surrogate1, surrogate2)

            # Compute KL divergence
            # kl = (ref_log_prob - log_prob) * torch.exp(ref_log_prob)

            # approximate KL divergence using the log probabilities
            kl = (ref_log_prob / log_prob) - torch.log(ref_log_prob / log_prob) - 1

            # Total loss = policy loss + KL penalty
            step_loss = policy_loss + self.beta * kl

            loss += step_loss
            kl_div += kl

        # Average losses over batch
        num_samples = max(len(states), 1)
        loss = loss / num_samples
        kl_div = kl_div / num_samples

        return loss, kl_div
