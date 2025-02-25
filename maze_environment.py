import numpy as np
import random

# Actions
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# Cell states
EMPTY = 0
WALL = 1
START = 2
GOAL = 3
AGENT = 4


class MazeEnvironment:
    """A grid world maze environment."""

    def __init__(self, size=10, wall_prob=0.2):
        """Initialize a random maze of given size."""
        self.size = size

        # Generate random maze
        self.maze = np.zeros((size, size), dtype=int)

        # Add walls randomly
        for i in range(size):
            for j in range(size):
                if random.random() < wall_prob:
                    self.maze[i, j] = WALL

        # Ensure start and goal are not walls
        valid_cells = [(i, j) for i in range(size) for j in range(size) if self.maze[i, j] != WALL]

        # Select start and goal positions
        self.start_pos, self.goal_pos = random.sample(valid_cells, 2)
        self.agent_pos = self.start_pos

        # Mark start and goal cells
        self.maze[self.start_pos] = START
        self.maze[self.goal_pos] = GOAL

        # Path history for visualization
        # format {"state": (row,col), "action": action, "reward": reward, "next_state": (row,col)}
        self.path_history = []

    def reset(self):
        """Reset agent to start position."""
        self.agent_pos = self.start_pos
        self.path_history = []  # Clear path history
        return self.agent_pos

    def step(self, action):
        """Take a step in the environment.

        action: one of UP, RIGHT, DOWN, LEFT

        returns: Return new state, reward, and done.
        """
        row, col = self.agent_pos
        old_pos = self.agent_pos
        actually_moved = False

        if action == UP and row > 0:  # Up
            new_pos = (row - 1, col)
        elif action == RIGHT and col < self.size - 1:  # Right
            new_pos = (row, col + 1)
        elif action == DOWN and row < self.size - 1:  # Down
            new_pos = (row + 1, col)
        elif action == LEFT and col > 0:  # Left
            new_pos = (row, col - 1)
        else:
            new_pos = self.agent_pos  # Invalid action (boundary case)
            print(f"Invalid move - boundary edge at position {self.agent_pos} with action {action}")

        # Check if the new position is a wall
        if new_pos != self.agent_pos and self.maze[new_pos] != WALL:
            self.agent_pos = new_pos
            actually_moved = True
        elif new_pos != self.agent_pos:
            print(f"Invalid move - wall collision at {new_pos}")

        # Determine if the goal is reached
        done = self.agent_pos == self.goal_pos
        reward = 1 if done else 0

        # Only add to path history if the agent actually moved
        if actually_moved:
            self.path_history.append(
                {"state": old_pos, "action": action, "reward": reward, "next_state": self.agent_pos}
            )

        return self.agent_pos, reward, done
