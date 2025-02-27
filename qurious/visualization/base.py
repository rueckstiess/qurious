import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches


class Layer(ABC):
    """Abstract base class for visualization layers."""

    def __init__(self, name: str = None, enabled: bool = True):
        """
        Initialize a visualization layer.

        Args:
            name: Optional name for the layer
            enabled: Whether the layer is initially enabled
        """
        self.name = name or self.__class__.__name__
        self.enabled = enabled

    @abstractmethod
    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the layer in ASCII format.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation
        """
        pass

    @abstractmethod
    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the layer using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        pass


class GridWorldVisualizer:
    """Base class for grid world visualization."""

    def __init__(self, env: Any):
        """
        Initialize the visualizer.

        Args:
            env: The grid world environment to visualize
        """
        self.env = env
        self.layers: List[Layer] = []

        # Default visualization settings
        self.config = {
            # ASCII representation
            "ascii_empty": ".",
            "ascii_obstacle": "#",
            "ascii_goal": "G",
            "ascii_agent": "A",
            # Matplotlib colors
            "color_empty": "white",
            "color_obstacle": "gray",
            "color_goal": "green",
            "color_agent": "red",
            # Figure settings
            "figsize": (10, 8),
            "dpi": 100,
            "cmap": None,  # Will be created based on colors
            "grid_alpha": 0.7,
            "text_fontsize": 9,
        }

        # Create color map
        self._update_colormap()

    def _update_colormap(self):
        """Update the colormap based on current colors."""
        self.config["cmap"] = ListedColormap(
            [
                self.config["color_empty"],
                self.config["color_obstacle"],
                self.config["color_goal"],
                self.config["color_agent"],
            ]
        )

    def add_layer(self, layer: Layer) -> "GridWorldVisualizer":
        """
        Add a visualization layer.

        Args:
            layer: The layer to add

        Returns:
            Self for method chaining
        """
        self.layers.append(layer)
        return self

    def get_layer(self, layer_name: str) -> Optional[Layer]:
        """
        Get a layer by name.

        Args:
            layer_name: Name of the layer to retrieve

        Returns:
            The layer if found, None otherwise
        """
        for layer in self.layers:
            if layer.name == layer_name:
                return layer
        return None

    def remove_layer(self, layer_name: str) -> bool:
        """
        Remove a layer by name.

        Args:
            layer_name: Name of the layer to remove

        Returns:
            True if the layer was removed, False otherwise
        """
        for i, layer in enumerate(self.layers):
            if layer.name == layer_name:
                self.layers.pop(i)
                return True
        return False

    def enable_layer(self, layer_name: str) -> bool:
        """Enable a layer by name."""
        layer = self.get_layer(layer_name)
        if layer:
            layer.enabled = True
            return True
        return False

    def disable_layer(self, layer_name: str) -> bool:
        """Disable a layer by name."""
        layer = self.get_layer(layer_name)
        if layer:
            layer.enabled = False
            return True
        return False

    def render_ascii(self) -> str:
        """
        Render the grid world as ASCII art.

        Returns:
            ASCII representation of the grid world
        """
        # Initialize grid with empty cells
        grid = np.full((self.env.height, self.env.width), self.config["ascii_empty"])

        # Apply all enabled layers
        for layer in self.layers:
            if layer.enabled:
                grid = layer.render_ascii(grid, self.env)

        # Convert to string
        result = ""
        for row in grid:
            result += " ".join(row) + "\n"

        return result

    def render_matplotlib(self, figsize=None, dpi=None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Render the grid world using matplotlib.

        Args:
            figsize: Optional figure size override
            dpi: Optional DPI override

        Returns:
            Figure and axes objects
        """
        # Create figure and axis
        figsize = figsize or self.config["figsize"]
        dpi = dpi or self.config["dpi"]
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Initialize grid for base elements
        grid = np.zeros((self.env.height, self.env.width))

        # Mark obstacles as 1
        for r, c in self.env.obstacles:
            if 0 <= r < self.env.height and 0 <= c < self.env.width:
                grid[r, c] = 1

        # Mark goals as 2
        for r, c in self.env.goal_pos:
            if 0 <= r < self.env.height and 0 <= c < self.env.width:
                grid[r, c] = 2

        # Mark agent position as 3 if available
        if hasattr(self.env, "position") and self.env.position is not None:
            r, c = self.env.position
            if 0 <= r < self.env.height and 0 <= c < self.env.width:
                grid[r, c] = 3

        # Plot the base grid
        ax.imshow(grid, cmap=self.config["cmap"])

        # Add grid lines
        for i in range(self.env.width + 1):
            ax.axvline(i - 0.5, color="black", linewidth=1, alpha=self.config["grid_alpha"])
        for i in range(self.env.height + 1):
            ax.axhline(i - 0.5, color="black", linewidth=1, alpha=self.config["grid_alpha"])

        # Apply all enabled layers
        for layer in self.layers:
            if layer.enabled:
                layer.render_matplotlib(fig, ax, grid, self.env)

        # Add labels and set ticks
        ax.set_xticks(range(self.env.width))
        ax.set_yticks(range(self.env.height))
        ax.set_xticklabels(range(self.env.width))
        ax.set_yticklabels(range(self.env.height))

        # Create legend
        legend_elements = [
            patches.Patch(facecolor=self.config["color_empty"], edgecolor="black", label="Empty"),
            patches.Patch(facecolor=self.config["color_obstacle"], edgecolor="black", label="Obstacle"),
            patches.Patch(facecolor=self.config["color_goal"], edgecolor="black", label="Goal"),
            patches.Patch(facecolor=self.config["color_agent"], edgecolor="black", label="Agent"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

        plt.tight_layout()
        return fig, ax

    def configure(self, **kwargs) -> "GridWorldVisualizer":
        """
        Update visualization configuration.

        Args:
            **kwargs: Configuration key-value pairs to update

        Returns:
            Self for method chaining
        """
        self.config.update(kwargs)

        # If any color was changed, update the colormap
        if any(k.startswith("color_") for k in kwargs):
            self._update_colormap()

        return self

    def savefig(self, filepath: str, **kwargs) -> None:
        """
        Save the current matplotlib visualization to a file.

        Args:
            filepath: Path to save the figure
            **kwargs: Additional arguments to pass to plt.savefig
        """
        fig, _ = self.render_matplotlib()
        fig.savefig(filepath, **kwargs)
        plt.close(fig)
