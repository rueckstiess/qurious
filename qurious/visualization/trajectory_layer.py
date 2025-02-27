import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
from typing import Any, Dict, Optional, List, Tuple, Union

from qurious.visualization.base import Layer
from qurious.visualization.grid_agent_layers import AgentLayer
from qurious.experience import Transition


class TrajectoryLayer(Layer):
    """Layer for rendering agent trajectories and paths."""

    def __init__(
        self,
        transitions=None,
        name: str = "Trajectory",
        enabled: bool = True,
        color: str = "blue",
        linewidth: float = 2.0,
        alpha: float = 0.7,
        show_start_end: bool = True,
        show_rewards: bool = False,
        start_color: str = "green",
        end_color: str = "red",
        marker_size: int = 100,
    ):
        """
        Initialize the trajectory layer.

        Args:
            transitions: List of transitions or experience object
            name: Layer name
            enabled: Whether the layer is enabled
            color: Color of the trajectory line
            linewidth: Width of the trajectory line
            alpha: Transparency of the trajectory
            show_start_end: Whether to highlight start and end points
            show_rewards: Whether to show rewards along the trajectory
            start_color: Color for the start marker
            end_color: Color for the end marker
            marker_size: Size of start/end markers
        """
        super().__init__(name, enabled)
        self.transitions = transitions if transitions is not None else []
        self.color = color
        self.linewidth = linewidth
        self.alpha = alpha
        self.show_start_end = show_start_end
        self.show_rewards = show_rewards
        self.start_color = start_color
        self.end_color = end_color
        self.marker_size = marker_size

    def set_transitions(self, transitions):
        """
        Set the transitions to visualize.

        Args:
            transitions: List of transitions or experience object
        """
        self.transitions = transitions

    def render_ascii(self, grid: np.ndarray, env: Any) -> np.ndarray:
        """
        Render the trajectory in ASCII.

        Args:
            grid: The current grid representation
            env: The grid world environment

        Returns:
            Updated grid representation with trajectory information
        """
        # Skip if no transitions
        if not self.transitions:
            return grid

        # Create a copy of the grid that we can modify
        trajectory_grid = grid.copy()

        # Get the states from transitions
        states = []
        for t in self.transitions:
            if hasattr(t, "state"):  # If it's a Transition object
                states.append(t.state)
            else:  # If it's a tuple (state, action, reward, next_state, done)
                states.append(t[0])

        # Mark trajectory cells with '*'
        for state in states:
            try:
                pos = env.index_to_state(state)
                row, col = pos

                # Only override empty cells or agent position
                if grid[row, col] in [".", "A"]:
                    trajectory_grid[row, col] = "*"
            except Exception:
                # Skip if error
                pass

        # Mark start and end if requested
        if self.show_start_end and states:
            # Start position
            try:
                start_pos = env.index_to_state(states[0])
                row, col = start_pos
                trajectory_grid[row, col] = "S"
            except Exception:
                pass

            # End position
            try:
                end_pos = env.index_to_state(states[-1])
                row, col = end_pos
                trajectory_grid[row, col] = "E"
            except Exception:
                pass

        return trajectory_grid

    def render_matplotlib(self, fig: plt.Figure, ax: plt.Axes, grid: np.ndarray, env: Any) -> None:
        """
        Render the trajectory using matplotlib.

        Args:
            fig: The matplotlib figure
            ax: The matplotlib axes
            grid: The current grid representation
            env: The grid world environment
        """
        # Skip if no transitions
        if not self.transitions:
            return

        # Extract state sequence
        states = []
        rewards = []

        for t in self.transitions:
            if hasattr(t, "state"):  # If it's a Transition object
                states.append(t.state)
                rewards.append(t.reward)
            else:  # If it's a tuple (state, action, reward, next_state, done)
                states.append(t[0])
                rewards.append(t[2])

        # Convert states to positions
        positions = []
        for state in states:
            try:
                pos = env.index_to_state(state)
                positions.append(pos)
            except Exception:
                # Skip if error
                pass

        if not positions:
            return

        # Plot the trajectory path
        # Convert positions to plot coordinates (center of cells)
        x_coords = [p[1] for p in positions]  # Column indices for x
        y_coords = [p[0] for p in positions]  # Row indices for y

        # Plot the path
        ax.plot(
            x_coords, y_coords, "-", color=self.color, linewidth=self.linewidth, alpha=self.alpha, zorder=10
        )  # Higher zorder to draw on top

        # Mark start and end points if requested
        if self.show_start_end and positions:
            # Start point
            ax.scatter(
                x_coords[0],
                y_coords[0],
                s=self.marker_size,
                c=self.start_color,
                marker="o",
                edgecolors="black",
                zorder=11,
                label="Start",
            )

            # End point
            ax.scatter(
                x_coords[-1],
                y_coords[-1],
                s=self.marker_size,
                c=self.end_color,
                marker="o",
                edgecolors="black",
                zorder=11,
                label="End",
            )

        # Show rewards if requested
        if self.show_rewards and rewards:
            for i, (x, y, r) in enumerate(zip(x_coords, y_coords, rewards)):
                # Only show non-zero rewards or first/last
                if r != 0 or i == 0 or i == len(rewards) - 1:
                    ax.text(
                        x,
                        y,
                        f"{r:+.1f}",
                        ha="center",
                        va="bottom",
                        color="black",
                        fontweight="bold",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                    )


class EpisodeAnimator:
    """Helper class for creating animated visualizations of episodes."""

    def __init__(
        self,
        env,
        visualizer,
        transitions=None,
        interval=500,  # ms between frames
        repeat_delay=1000,  # ms before repeating
    ):
        """
        Initialize the episode animator.

        Args:
            env: The grid world environment
            visualizer: The GridWorldVisualizer instance
            transitions: List of transitions or experience object
            interval: Milliseconds between frames
            repeat_delay: Milliseconds before repeating animation
        """
        self.env = env
        self.visualizer = visualizer
        self.transitions = transitions if transitions is not None else []
        self.interval = interval
        self.repeat_delay = repeat_delay

        # Get the initial state of the environment
        self.initial_state = None
        if hasattr(env, "position"):
            self.initial_state = env.position

    def set_transitions(self, transitions):
        """Set the transitions to animate."""
        self.transitions = transitions

    def animate(self, save_path=None, fps=5, dpi=100, display_in_notebook=False):
        """
        Create and optionally save an animation of the episode.

        Args:
            save_path: If provided, save the animation to this path
            fps: Frames per second for saved animation
            dpi: DPI for saved animation
            display_in_notebook: Whether to display animation in a Jupyter notebook

        Returns:
            The matplotlib animation object
        """
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation

        if not self.transitions:
            return None

        # Extract state sequence
        states = []
        for t in self.transitions:
            if hasattr(t, "state"):  # If it's a Transition object
                states.append(t.state)
            else:  # If it's a tuple (state, action, reward, next_state, done)
                states.append(t[0])

        # Create figure and axes for animation
        fig, ax = plt.subplots(figsize=self.visualizer.config["figsize"], dpi=self.visualizer.config["dpi"])

        # Create trajectory layer for path history
        trajectory_layer = TrajectoryLayer(enabled=True, alpha=0.3)
        self.visualizer.add_layer(trajectory_layer)

        # Create agent layer that will be updated
        agent_layer = AgentLayer(enabled=True, position=None)
        self.visualizer.add_layer(agent_layer)

        # Function to update the animation for each frame
        def update(frame):
            # Clear the axis
            ax.clear()

            # Update agent position
            if frame < len(states):
                state = states[frame]
                position = self.env.index_to_state(state)
                agent_layer.position = position

                # Update trajectory with states up to current frame
                current_transitions = self.transitions[: frame + 1]
                trajectory_layer.set_transitions(current_transitions)

            # Render the visualization directly to the existing fig and ax
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
            if agent_layer.position is not None:
                r, c = agent_layer.position
                if 0 <= r < self.env.height and 0 <= c < self.env.width:
                    grid[r, c] = 3

            # Plot the base grid
            ax.imshow(grid, cmap=self.visualizer.config["cmap"])

            # Add grid lines
            for i in range(self.env.width + 1):
                ax.axvline(i - 0.5, color="black", linewidth=1, alpha=self.visualizer.config["grid_alpha"])
            for i in range(self.env.height + 1):
                ax.axhline(i - 0.5, color="black", linewidth=1, alpha=self.visualizer.config["grid_alpha"])

            # Apply all enabled layers
            for layer in self.visualizer.layers:
                if layer.enabled:
                    layer.render_matplotlib(fig, ax, grid, self.env)

            # Add labels and set ticks
            ax.set_xticks(range(self.env.width))
            ax.set_yticks(range(self.env.height))
            ax.set_xticklabels(range(self.env.width))
            ax.set_yticklabels(range(self.env.height))

            # Add a frame counter
            ax.set_title(f"Frame {frame + 1}/{len(states)}")

            return (ax,)

        # Create the animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(states),
            interval=self.interval,
            blit=False,
            repeat=True,
            repeat_delay=self.repeat_delay,
        )

        # Save if requested
        if save_path:
            writer = animation.PillowWriter(fps=fps)
            anim.save(save_path, writer=writer, dpi=dpi)

        # If in a notebook environment and display is requested, display it
        if display_in_notebook:
            try:
                from IPython.display import display, HTML

                # Create HTML animation
                html = anim.to_jshtml()
                # Close the original figure to prevent duplicate display
                plt.close(fig)
                # Return HTML that will be displayed in the notebook
                return HTML(html)
            except ImportError:
                print("IPython not available. Cannot display in notebook.")
                # Keep the figure open if we can't display in notebook
                return anim

        # Close the figure if we're not displaying in notebook
        if not display_in_notebook:
            plt.close(fig)

        return anim
