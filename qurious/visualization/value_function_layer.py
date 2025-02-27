import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Any, Dict, Optional, List, Tuple, Union

from qurious.visualization.base import Layer


class ValueFunctionLayer(Layer):
    """Layer for rendering state value functions as a heatmap."""
    
    def __init__(
        self, 
        value_function, 
        name: str = "ValueFunction", 
        enabled: bool = True,
        cmap: str = 'viridis',
        alpha: float = 0.5,
        show_text: bool = True,
        text_color: str = 'black',
        text_fontsize: int = 9,
        text_precision: int = 2,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None
    ):
        """
        Initialize the value function layer.
        
        Args:
            value_function: The value function to visualize
            name: Layer name
            enabled: Whether the layer is enabled
            cmap: Colormap for the heatmap
            alpha: Transparency of the heatmap overlay
            show_text: Whether to show value text
            text_color: Color of the text
            text_fontsize: Font size for the text
            text_precision: Decimal precision for displayed values
            vmin: Minimum value for colormap scaling (auto if None)
            vmax: Maximum value for colormap scaling (auto if None)
        """
        super().__init__(name, enabled)
        self.value_function = value_function
        self.cmap = cmap
        self.alpha = alpha
        self.show_text = show_text
        self.text_color = text_color
        self.text_fontsize = text_fontsize
        self.text_precision = text_precision
        self.vmin = vmin
        self.vmax = vmax
    
    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the value function in ASCII.
        
        Args:
            grid: The current grid representation
            env: The grid world environment
            
        Returns:
            Updated grid representation with value function information
        """
        # For ASCII, we can't really show a heatmap, so we'll just return the grid
        return grid
    
    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the value function using matplotlib.
        
        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        # Create a value grid
        value_grid = np.zeros((env.height, env.width))
        masked_grid = np.ma.masked_array(value_grid, mask=np.zeros_like(value_grid, dtype=bool))
        
        # Fill in values
        for row in range(env.height):
            for col in range(env.width):
                # Skip obstacles
                if grid[row, col] == 1:  # 1=obstacle in the numeric grid
                    masked_grid.mask[row, col] = True
                    continue
                
                # Get state index
                state = env.state_to_index((row, col))
                
                # Get value
                try:
                    if hasattr(self.value_function, 'estimate_all_actions'):
                        # For action-value functions, use maximum value across actions
                        value = np.max(self.value_function.estimate_all_actions(state))
                    else:
                        # For state-value functions
                        value = self.value_function.estimate(state)
                    
                    value_grid[row, col] = value
                except Exception as e:
                    # If there's an error estimating value, mask this cell
                    masked_grid.mask[row, col] = True
        
        # Determine colormap range
        actual_min = np.min(value_grid[~masked_grid.mask]) if np.any(~masked_grid.mask) else 0
        actual_max = np.max(value_grid[~masked_grid.mask]) if np.any(~masked_grid.mask) else 1
        
        vmin = self.vmin if self.vmin is not None else actual_min
        vmax = self.vmax if self.vmax is not None else actual_max
        
        # Create a custom norm to handle masked values
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot the heatmap
        im = ax.imshow(
            masked_grid, 
            cmap=self.cmap, 
            alpha=self.alpha, 
            norm=norm,
            interpolation='nearest'
        )
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Value')
        
        # Display values as text if requested
        if self.show_text:
            for row in range(env.height):
                for col in range(env.width):
                    if not masked_grid.mask[row, col]:
                        value = value_grid[row, col]
                        text = f"{value:.{self.text_precision}f}"
                        ax.text(
                            col, 
                            row, 
                            text, 
                            ha='center', 
                            va='center', 
                            color=self.text_color,
                            fontsize=self.text_fontsize,
                            fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
                        )
