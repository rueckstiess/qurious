from typing import Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from qurious.visualization.base import Layer


class ActionValueLayer(Layer):
    """Layer for rendering action-value functions (Q-values)."""

    def __init__(
        self,
        action_value_function,
        name: str = "ActionValue",
        enabled: bool = True,
        cmap: str = "viridis",
        alpha: float = 0.7,
        text_color: str = "black",
        text_fontsize: int = 8,
        text_precision: int = 2,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """
        Initialize the action-value function layer.

        Args:
            action_value_function: The action-value function to visualize
            name: Layer name
            enabled: Whether the layer is enabled
            cmap: Colormap for the values
            alpha: Transparency of the visualization
            text_color: Color of the text
            text_fontsize: Font size for the text
            text_precision: Decimal precision for displayed values
            vmin: Minimum value for colormap scaling (auto if None)
            vmax: Maximum value for colormap scaling (auto if None)
        """
        super().__init__(name, enabled)
        self.action_value_function = action_value_function
        self.cmap = cmap
        self.alpha = alpha
        self.text_color = text_color
        self.text_fontsize = text_fontsize
        self.text_precision = text_precision
        self.vmin = vmin
        self.vmax = vmax

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the action-value function in ASCII.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with action-value information
        """
        # For ASCII, we can't really show Q-values effectively, so we'll just return the grid
        return grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the action-value function using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        # Define action coordinates (relative to cell center)
        action_coords = [
            (0.5, 0.2),  # UP
            (0.8, 0.5),  # RIGHT
            (0.5, 0.8),  # DOWN
            (0.2, 0.5),  # LEFT
        ]

        # Collect all values to determine colormap range if not provided
        all_values = []

        # First pass to collect values
        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles
                if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                    continue

                # Get state index
                state = env.state_to_index((row, col))

                # Get values for all actions
                try:
                    q_values = self.action_value_function.estimate_all_actions(state)
                    all_values.extend(q_values)
                except Exception:
                    # Skip if error
                    pass

        # Determine colormap range
        if all_values:
            actual_min = min(all_values)
            actual_max = max(all_values)

            vmin = self.vmin if self.vmin is not None else actual_min
            vmax = self.vmax if self.vmax is not None else actual_max

            # Create colormap and normalizer
            cmap = plt.get_cmap(self.cmap)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Second pass to plot values
            for row in range(env.height):
                for col in range(env.width):
                    # Skip obstacles
                    if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                        continue

                    # Get state index
                    state = env.state_to_index((row, col))

                    try:
                        # Get values for all actions
                        q_values = self.action_value_function.estimate_all_actions(state)

                        # Plot Q-values for each action
                        for action, q_value in enumerate(q_values):
                            dx, dy = action_coords[action]

                            # Determine color based on value
                            color = cmap(norm(q_value))

                            # convert color to 50% transparency
                            # color = color[:3] + (0.5,)

                            # Create a small circle with color based on value
                            # circle = plt.Circle(
                            #     (col + dx - 0.5, row + dy - 0.5),
                            #     0.1,
                            #     color=color,
                            #     alpha=self.alpha
                            # )
                            # ax.add_patch(circle)

                            # rotate text for left and right action by 90 degrees
                            rotation = -90 if action in [1, 3] else 0

                            # Add text with the Q-value
                            ax.text(
                                col + dx - 0.5,
                                row + dy - 0.5,
                                f"{q_value:.{self.text_precision}f}",
                                ha="center",
                                va="center",
                                fontsize=self.text_fontsize,
                                color=self.text_color,
                                rotation=rotation,
                                bbox=dict(edgecolor="white", facecolor=color, alpha=0.5, pad=1),
                            )
                    except Exception:
                        # Skip if error
                        pass

            # Add a colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.8)
            cbar.set_label("Q-Value")
