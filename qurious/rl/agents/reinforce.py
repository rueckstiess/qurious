from typing import Dict, List, Tuple, Union

import numpy as np
import torch

from qurious.rl.experience import Transition

from .policy_gradient import PolicyGradientAgent


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
                G = torch.tensor(G, device=log_prob.device, dtype=torch.float32)

            # Policy gradient loss: -log(π(a|s)) * G
            step_loss = -log_prob * G
            loss += step_loss

        # Average loss over trajectory
        return loss / len(states)
