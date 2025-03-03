from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qurious.visualization.base import Layer


class PolicyLayer(Layer):
    """Layer for rendering policy information."""

    # Direction symbols for ASCII rendering
    ASCII_DIRS = ["↑", "→", "↓", "←"]

    def __init__(
        self,
        policy,
        name: str = "Policy",
        enabled: bool = True,
        arrow_scale: float = 1.0,
        threshold: float = 0.05,
        arrow_color: str = (0.8, 0.8, 0.8, 1),
        text_color: str = "black",
        show_text: bool = False,
    ):
        """
        Initialize the policy layer.

        Args:
            policy: The policy to visualize
            name: Layer name
            enabled: Whether the layer is enabled
            arrow_scale: Scale factor for arrow size
            threshold: Minimum probability to display an arrow
            color: Arrow color
            show_text: Whether to show probability text
        """
        super().__init__(name, enabled)
        self.policy = policy
        self.arrow_scale = arrow_scale
        self.threshold = threshold
        self.text_color = text_color
        self.arrow_color = arrow_color
        self.show_text = show_text

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the policy in ASCII.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with policy information
        """
        # For ASCII, we'll use a simplified representation
        # For deterministic policies, replace cells with arrow symbols
        # For stochastic policies, we'll use the highest probability action

        # Create a copy of the grid that we can modify
        policy_grid = grid.copy()

        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles and goals
                cell_content = grid[row, col]
                if cell_content in ["#", "G", "A"]:
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Get action probabilities
                action_probs = self.policy.get_action_probabilities(state)

                # Find the highest probability action
                max_action = np.argmax(action_probs)
                max_prob = action_probs[max_action]

                # Replace cell with direction if probability is significant
                if max_prob >= self.threshold:
                    policy_grid[row, col] = self.ASCII_DIRS[max_action]

        return policy_grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the policy using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        # Direction vectors for actions [UP, RIGHT, DOWN, LEFT]
        action_dirs = [
            (0, -0.4),  # UP: no x change, negative y (up in plot)
            (0.4, 0),  # RIGHT: positive x, no y change
            (0, 0.4),  # DOWN: no x change, positive y (down in plot)
            (-0.4, 0),  # LEFT: negative x, no y change
        ]

        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles and goal cells
                if grid[row, col] in [1, 2]:  # 1=obstacle, 2=goal in the numeric grid
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Get action probabilities
                action_probs = self.policy.get_action_probabilities(state)

                # Find max probability for scaling
                max_prob = max(action_probs)

                # Plot arrows for each action with length proportional to probability
                for action, prob in enumerate(action_probs):
                    if prob > self.threshold:  # Only plot if probability is significant
                        # Scale arrow by relative probability (compared to max)
                        scale = 0.2 + 0.8 * (prob / max_prob) if max_prob > 0 else 0
                        scale *= self.arrow_scale  # Apply user-defined scaling

                        # Get direction vector and scale it
                        dx, dy = action_dirs[action]
                        dx *= scale
                        dy *= scale

                        # Draw arrow
                        ax.arrow(
                            col,
                            row,  # Start position (center of cell)
                            dx,
                            dy,  # Arrow direction and length
                            head_width=0.15 * scale,
                            head_length=0.15 * scale,
                            fc=self.arrow_color,
                            ec=self.arrow_color,
                            length_includes_head=True,
                        )

                        # Optionally show probability text
                        if self.show_text and prob >= self.threshold:  # Only show for significant probabilities
                            # Position the text near the arrow
                            text_pos = (col + np.sign(dx) * 0.3, row + np.sign(dy) * 0.4)
                            ax.text(
                                text_pos[0],
                                text_pos[1],
                                f"{prob:.2f}",
                                ha="center",
                                va="center",
                                fontsize=6,
                                color=self.text_color,
                            )
