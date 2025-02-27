import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Optional

from qurious.visualization.base import Layer


class EligibilityTraceLayer(Layer):
    """Layer for rendering eligibility traces as a heatmap."""

    def __init__(
        self,
        eligibility_traces=None,
        name: str = "EligibilityTrace",
        enabled: bool = True,
        cmap: str = "hot",
        alpha: float = 0.5,
        show_text: bool = True,
        text_color: str = "black",
        text_fontsize: int = 8,
        text_precision: int = 2,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ):
        """
        Initialize the eligibility trace layer.

        Args:
            eligibility_traces: Dictionary mapping state IDs to eligibility values or array with same dimensions as state space
            name: Layer name
            enabled: Whether the layer is enabled
            cmap: Colormap for the heatmap
            alpha: Transparency of the heatmap overlay
            show_text: Whether to show trace values as text
            text_color: Color of the text
            text_fontsize: Font size for the text
            text_precision: Decimal precision for displayed values
            vmin: Minimum value for colormap scaling (auto if None)
            vmax: Maximum value for colormap scaling (auto if None)
        """
        super().__init__(name, enabled)
        self.eligibility_traces = eligibility_traces if eligibility_traces is not None else {}
        self.cmap = cmap
        self.alpha = alpha
        self.show_text = show_text
        self.text_color = text_color
        self.text_fontsize = text_fontsize
        self.text_precision = text_precision
        self.vmin = vmin
        self.vmax = vmax

    def set_traces(self, eligibility_traces):
        """Set the eligibility traces to visualize."""
        self.eligibility_traces = eligibility_traces

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the eligibility traces in ASCII.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with eligibility trace information
        """
        # For ASCII, we can't show the heatmap effectively
        # Return unmodified grid
        return grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the eligibility traces using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        # Skip rendering if traces are empty
        if isinstance(self.eligibility_traces, dict) and not self.eligibility_traces:
            return
        elif isinstance(self.eligibility_traces, np.ndarray) and (
            self.eligibility_traces.size == 0 or np.all(self.eligibility_traces == 0)
        ):
            return

        # Create a grid for eligibility values
        trace_grid = np.zeros((env.height, env.width))
        masked_grid = np.ma.masked_array(trace_grid, mask=np.zeros_like(trace_grid, dtype=bool))

        # Fill in eligibility values
        if isinstance(self.eligibility_traces, dict):
            # If traces are stored as a dictionary mapping state IDs to values
            for state, trace_value in self.eligibility_traces.items():
                try:
                    row, col = env.index_to_state(state)
                    # Skip obstacles
                    if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                        masked_grid.mask[row, col] = True
                        continue

                    trace_grid[row, col] = trace_value
                except Exception:
                    # Skip if error
                    pass
        elif isinstance(self.eligibility_traces, np.ndarray):
            # If traces are stored as a 1D or 2D array
            if self.eligibility_traces.ndim == 1:
                # 1D array with values for each state
                for state, trace_value in enumerate(self.eligibility_traces):
                    try:
                        row, col = env.index_to_state(state)
                        # Skip obstacles
                        if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                            masked_grid.mask[row, col] = True
                            continue

                        trace_grid[row, col] = trace_value
                    except Exception:
                        # Skip if error
                        pass
            elif self.eligibility_traces.ndim == 2:
                # Assume it's already in grid format
                trace_height, trace_width = self.eligibility_traces.shape
                for row in range(min(env.height, trace_height)):
                    for col in range(min(env.width, trace_width)):
                        # Skip obstacles
                        if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                            masked_grid.mask[row, col] = True
                            continue

                        trace_grid[row, col] = self.eligibility_traces[row, col]

        # Determine colormap range
        nonzero_values = trace_grid[~masked_grid.mask & (trace_grid != 0)]
        if len(nonzero_values) > 0:
            actual_min = np.min(nonzero_values)
            actual_max = np.max(nonzero_values)

            vmin = self.vmin if self.vmin is not None else actual_min
            vmax = self.vmax if self.vmax is not None else actual_max

            # Create a custom norm to handle masked values
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            # Plot the heatmap
            im = ax.imshow(masked_grid, cmap=self.cmap, alpha=self.alpha, norm=norm, interpolation="nearest")

            # Add a colorbar
            cbar = fig.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Eligibility Trace")

            # Display values as text if requested
            if self.show_text:
                for row in range(env.height):
                    for col in range(env.width):
                        if not masked_grid.mask[row, col] and trace_grid[row, col] > 0:
                            value = trace_grid[row, col]
                            text = f"{value:.{self.text_precision}f}"
                            ax.text(
                                col,
                                row,
                                text,
                                ha="center",
                                va="center",
                                color=self.text_color,
                                fontsize=self.text_fontsize,
                                fontweight="bold",
                                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
                            )
