import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from typing import Any, Optional, Tuple

from qurious.visualization.base import Layer


class GridLayer(Layer):
    """Layer for rendering the basic grid with obstacles and goals."""

    def __init__(self, name: str = "Grid", enabled: bool = True):
        """Initialize the grid layer."""
        super().__init__(name, enabled)

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the basic grid structure in ASCII.

        Args:
            grid: The starting grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with obstacles and goals
        """
        # The base grid already has empty cells

        # Mark obstacles
        for r, c in env.obstacles:
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = "#"

        # Mark goals
        for r, c in env.goal_pos:
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = "G"

        return grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the grid using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """

        # Initialize grid for base elements
        # grid = np.zeros((env.height, env.width))

        # Mark obstacles as 2
        for r, c in env.obstacles:
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = 1  # Mark obstacles as 1

        # Mark goals as 3
        for r, c in env.goal_pos:
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = 2  # Mark goals as 2

        # Get existing legend handles and labels
        handles, _ = ax.get_legend().legend_handles, ax.get_legend().get_texts()

        # Create agent patch
        extra_patches = [
            patches.Patch(
                facecolor=self.config["color_obstacle"], edgecolor=self.config["color_obstacle"], label="Obstacle"
            ),
            patches.Patch(facecolor=self.config["color_goal"], edgecolor=self.config["color_obstacle"], label="Goal"),
        ]

        # Add new patch to existing handles and update legend
        handles.extend(extra_patches)
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=len(handles),
            frameon=False,
            facecolor=self.config["color_empty"],
            edgecolor=self.config["color_obstacle"],
            fontsize=self.config["text_fontsize"],
            labelcolor=self.config["color_obstacle"],
        )


class AgentLayer(Layer):
    """Layer for rendering the agent position."""

    def __init__(self, name: str = "Agent", enabled: bool = True, position: Optional[Tuple[int, int]] = None):
        """
        Initialize the agent layer.

        Args:
            name: Layer name
            enabled: Whether the layer is enabled
            position: Optional fixed position override (if None, uses environment's position)
        """
        super().__init__(name, enabled)
        self.position = position

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the agent position in ASCII.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with agent
        """
        # Use provided position or get from environment
        pos = self.position or env.position

        if pos is not None:
            r, c = pos
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = "A"

        return grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the agent using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """

        # Use provided position or get from environment
        pos = self.position or env.position
        if pos is not None:
            r, c = pos
            if 0 <= r < env.height and 0 <= c < env.width:
                grid[r, c] = 3

        # Get existing legend handles and labels
        handles, _ = ax.get_legend().legend_handles, ax.get_legend().get_texts()

        # Create agent patch
        agent_patch = patches.Patch(
            facecolor=self.config["color_agent"], edgecolor=self.config["color_obstacle"], label="Agent"
        )

        # Add new patch to existing handles and update legend
        handles.append(agent_patch)
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncols=len(handles),
            frameon=False,
            facecolor=self.config["color_empty"],
            edgecolor=self.config["color_obstacle"],
            fontsize=self.config["text_fontsize"],
            labelcolor=self.config["color_obstacle"],
        )
