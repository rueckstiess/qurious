from typing import Callable, Optional

import torch
from torch import nn

from qurious.rl.policies.policy import DiscretePolicy
from qurious.utils import auto_device


class DiscreteNNPolicy(DiscretePolicy):
    def __init__(
        self,
        input_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        feature_extractor: Optional[Callable] = None,
        device: str = "auto",
    ):
        """
        Initialize the neural network policy.
        Args:
            input_dim (int): Dimension of the input state.
            n_actions (int): Number of possible actions.
            hidden_dim (int): Dimension of the hidden layer.
        """
        super().__init__(n_actions)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.feature_extractor = feature_extractor

        # create model, loss and optimizer
        self.device = auto_device() if device == "auto" else device
        self.model = self._create_model()
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def _create_model(self):
        """
        Create the neural network model.

        Returns:
            torch.nn.Module: The neural network model.
        """
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions),
            nn.Softmax(dim=-1),
        )

    def get_action(self, state):
        """
        Get an action for the given state.

        Args:
            state: The current state

        Returns:
            An action to take
        """
        action_probs = self.get_action_probabilities(state)
        return action_probs.argmax()

    def get_action_probabilities(self, state):
        """
        Get probability distribution over actions for the given state.

        Args:
            state: The current state

        Returns:
            numpy.ndarray: Probability distribution over actions
        """

        features = self.feature_extractor(state) if self.feature_extractor else state
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs = self.model(features)

        action_probs = action_probs.cpu().numpy()
        return action_probs

    def update(self, state, action, value=None):
        pass

    def update_from_value_fn(self, value_function):
        pass
