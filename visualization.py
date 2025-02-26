import numpy as np
import matplotlib.pyplot as plt
from maze_environment import UP, DOWN, LEFT, RIGHT, EMPTY, WALL, START, GOAL, AGENT

# Colors for visualization
COLORS = {
    EMPTY: (1.0, 1.0, 1.0, 0.0),
    WALL: (0.0, 0.0, 0.0, 1.0),
    START: (0.0, 1.0, 0.0, 1.0),
    GOAL: (1.0, 0.0, 0.0, 1.0),
    AGENT: (0.0, 0.0, 1.0, 1.0),
}


class MazeVisualizer:
    """Visualizer for the maze environment."""

    def __init__(self, env, custom_colors=None, custom_arrow_style=None):
        """
        Initialize visualizer with a maze environment.

        Args:
            env: MazeEnvironment instance
            custom_colors: Optional dict to override default COLORS
            custom_arrow_style: Optional dict to override default arrow styling
        """
        self.env = env
        self.colors = COLORS.copy()

        # Override default colors if provided
        if custom_colors:
            for cell_type, color in custom_colors.items():
                self.colors[cell_type] = color

        # Setup the figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Set figure with transparent background
        self.fig.patch.set_facecolor("white")
        self.fig.patch.set_alpha(0.0)
        self.ax.set_facecolor("white")
        self.ax.patch.set_alpha(0.0)

        # Arrow styling parameters
        self.arrow_color = "black"
        self.arrow_width = 0.02
        self.arrow_head_width = 0.15
        self.arrow_head_length = 0.15
        self.arrow_alpha = 0.8
        self.arrow_offset = 0.25  # Distance from center to start/end of arrow (0.25 = quarter of cell width)

        # Override default arrow styling if provided
        if custom_arrow_style:
            for key, value in custom_arrow_style.items():
                setattr(self, key, value)

    def update_display(self, fig=None, ax=None):
        """
        Update the display based on current environment state

        Args:
            fig: Optional figure to use instead of self.fig
            ax: Optional axes to use instead of self.ax
        """
        # Use provided fig/ax or defaults
        fig = fig or self.fig
        ax = ax or self.ax

        ax.clear()

        # Create a copy of the maze for visualization
        vis_maze = self.env.maze.copy()

        # Mark agent position
        vis_maze[self.env.agent_pos] = AGENT

        # Create RGBA grid for visualization
        rgba_grid = np.ones((self.env.size, self.env.size, 4))

        for i in range(self.env.size):
            for j in range(self.env.size):
                cell_type = vis_maze[i, j]
                if cell_type in self.colors:
                    rgba_grid[i, j] = self.colors[cell_type]

        print(rgba_grid)
        # Show grid with RGBA values
        ax.imshow(rgba_grid)

        # Add text labels for START and GOAL positions
        start_row, start_col = self.env.start_pos
        goal_row, goal_col = self.env.goal_pos

        # Add START text
        ax.text(
            start_col,
            start_row,
            "START",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="normal",
        )

        # Add GOAL text
        ax.text(
            goal_col,
            goal_row,
            "GOAL",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=12,
            fontweight="normal",
        )

        # Draw path arrows if there is a path history
        if self.env.path_history:
            self.draw_path_arrows(ax=ax)

        # Add grid lines
        for i in range(self.env.size + 1):
            ax.axhline(i - 0.5, color="gray", linewidth=1)
            ax.axvline(i - 0.5, color="gray", linewidth=1)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title showing the current step
        ax.set_title(f"Step {len(self.env.path_history)}")

        fig.canvas.draw_idle()
        plt.pause(0.001) if fig is self.fig else None  # Only pause for interactive display

    def draw_path_arrows(self, path_history=None, ax=None):
        """
        Draw arrows showing the path taken by the agent.

        Args:
            path_history: Optional path history to use instead of self.env.path_history
            ax: Optional axes to use instead of self.ax
        """
        # Use provided axes or default
        ax = ax or self.ax

        # Use provided path history or default
        path_history = path_history or self.env.path_history

        # Draw arrows for each entry in path_history
        for i in range(len(path_history)):
            entry = path_history[i]
            action = entry["action"]
            row, col = entry["state"]

            # Calculate start point (offset from center toward edge)
            if action == UP:
                start_x, start_y = col, row - self.arrow_offset
                # Calculate destination (next cell)
                dest_row, dest_col = row - 1, col
            elif action == RIGHT:
                start_x, start_y = col + self.arrow_offset, row
                dest_row, dest_col = row, col + 1
            elif action == DOWN:
                start_x, start_y = col, row + self.arrow_offset
                dest_row, dest_col = row + 1, col
            elif action == LEFT:
                start_x, start_y = col - self.arrow_offset, row
                dest_row, dest_col = row, col - 1
            else:
                continue  # Skip invalid actions

            # Calculate end point (quarter cell from center of destination cell)
            if action == UP:
                end_x, end_y = dest_col, dest_row + self.arrow_offset
            elif action == RIGHT:
                end_x, end_y = dest_col - self.arrow_offset, dest_row
            elif action == DOWN:
                end_x, end_y = dest_col, dest_row - self.arrow_offset
            elif action == LEFT:
                end_x, end_y = dest_col + self.arrow_offset, dest_row

            # Calculate dx, dy for the arrow
            dx = end_x - start_x
            dy = end_y - start_y

            # Draw arrow
            ax.arrow(
                start_x,
                start_y,
                dx,
                dy,
                head_width=self.arrow_head_width,
                head_length=self.arrow_head_length,
                fc=self.arrow_color,
                ec=self.arrow_color,
                width=self.arrow_width,
                length_includes_head=True,
                alpha=self.arrow_alpha,
            )

    def create_animation(self, filename="maze_animation.gif", fps=2, end_pause_seconds=3):
        """
        Create and save an animation of the agent's path through the maze.

        Args:
            filename: Output filename for the GIF
            fps: Frames per second for the animation
            end_pause_seconds: Number of seconds to pause at the end of the animation

        Returns:
            None
        """
        # Check if we have a path to animate
        if not self.env.path_history:
            print("No path history to animate!")
            return None

        # Store the original path history and agent position
        original_path_history = self.env.path_history.copy()
        original_agent_pos = self.env.agent_pos

        # Use PIL directly for creating the GIF
        try:
            import io
            from PIL import Image
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

            # Create a new figure for the animation to avoid interference with interactive display
            fig, ax = plt.subplots(figsize=(8, 8))

            # List to store image frames
            frames = []

            # Generate frames for each step in the path history
            for frame_idx in range(len(original_path_history) + 1):
                # Clear axes for new frame
                ax.clear()

                # Set agent position and path history for this frame
                if frame_idx == 0:
                    # Initial state - agent at start position
                    self.env.agent_pos = self.env.start_pos
                    self.env.path_history = []
                else:
                    # Path history up to this frame
                    self.env.path_history = original_path_history[:frame_idx]
                    # Update agent position based on the last entry in the path history
                    self.env.agent_pos = original_path_history[frame_idx - 1]["next_state"]

                # Update display using our existing method with the animation figure/axes
                self.update_display(fig=fig, ax=ax)

                # Convert to PIL Image
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
                buf.seek(0)
                img = Image.open(buf)
                frames.append(img.copy())  # Add a copy to frames
                buf.close()

            # Add pause frames at the end by duplicating the last frame
            if frames and end_pause_seconds > 0:
                last_frame = frames[-1]
                pause_frames_count = int(fps * end_pause_seconds)
                for _ in range(pause_frames_count):
                    frames.append(last_frame.copy())

            # Close the figure to avoid displaying it
            plt.close(fig)

            # Save as GIF if we have frames
            if frames:
                frames[0].save(
                    filename, save_all=True, append_images=frames[1:], optimize=False, duration=int(1000 / fps), loop=0
                )
                print(f"Animation saved to {filename} with {end_pause_seconds} second pause at the end")
            else:
                print("No frames were generated for the animation")

        except Exception as e:
            print(f"Error creating animation: {e}")

        finally:
            # Restore original path history and agent position
            self.env.path_history = original_path_history
            self.env.agent_pos = original_agent_pos

        return None
