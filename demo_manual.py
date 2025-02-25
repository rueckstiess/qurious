from maze_environment import MazeEnvironment, UP, DOWN, LEFT, RIGHT, EMPTY, WALL, START, GOAL, AGENT
from visualization import MazeVisualizer
from agents import ManualAgent

# Example custom colors with RGBA values
custom_colors = {
    EMPTY: (1.0, 1.0, 1.0, 0.0),  # Transparent
    WALL: (0.1, 0.1, 0.1, 1.0),  # Nearly black
    START: (0.0, 0.7, 0.3, 0.8),  # Custom green with alpha
    GOAL: (0.9, 0.2, 0.2, 0.8),  # Custom red with alpha
    AGENT: (0.2, 0.4, 0.8, 1.0),  # Custom blue, fully opaque
}


def manual_control_demo():
    """Run a demo with manual agent control."""
    # Create environment
    env = MazeEnvironment(size=6, wall_prob=0.2)

    # Create visualizer with custom colors and arrow style
    visualizer = MazeVisualizer(env)

    # Create manual agent
    agent = ManualAgent(env, visualizer)

    # Run an episode
    agent.run_episode()

    # Save animation of the path
    visualizer.create_animation(filename="manual_path.gif", fps=5)

    print("Manual agent demo complete!")


if __name__ == "__main__":
    manual_control_demo()
