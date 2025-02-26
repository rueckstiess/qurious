import numpy as np
from mini_rl.mdp import MarkovDecisionProcess
from .environment import Environment


class GridWorld(Environment):
    """
    A simple grid world environment.

    The agent can move in four directions: up, down, left, right.
    Some cells may be obstacles (walls), and others may be terminal states (goals).
    """

    # Action indices
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(
        self,
        width=5,
        height=5,
        start_pos=(0, 0),
        goal_pos=None,
        obstacles=None,
        terminal_reward=1.0,
        step_penalty=0.0,
        max_steps=100,
    ):
        """
        Initialize a grid world.

        Args:
            width (int): Width of the grid
            height (int): Height of the grid
            start_pos (tuple): Starting position (row, column)
            goal_pos (list): List of goal positions [(row, column), ...]
            obstacles (list): List of obstacle positions [(row, column), ...]
            terminal_reward (float): Reward for reaching a goal
            step_penalty (float): Penalty for each step taken
            max_steps (int): Maximum steps before episode terminates
        """
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos if goal_pos is not None else [(height - 1, width - 1)]
        self.obstacles = obstacles if obstacles is not None else []
        self.terminal_reward = terminal_reward
        self.step_penalty = step_penalty
        self.max_steps = max_steps
        self._done = False

        # State representation
        self.position = None
        self.step_count = 0

        # Reset to initialize state
        self.reset()

    def reset(self):
        """
        Reset the environment to the initial state.

        Returns:
            State representation (position index)
        """
        self.position = self.start_pos
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        """
        Get the current state as an index.

        Returns:
            int: State index
        """
        return self.state_to_index(self.position)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): Action to take (UP, RIGHT, DOWN, LEFT)

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Get current position
        row, col = self.position

        # Calculate new position based on action
        if action == self.UP:
            new_row, new_col = max(0, row - 1), col
        elif action == self.RIGHT:
            new_row, new_col = row, min(self.width - 1, col + 1)
        elif action == self.DOWN:
            new_row, new_col = min(self.height - 1, row + 1), col
        elif action == self.LEFT:
            new_row, new_col = row, max(0, col - 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if the new position is valid (not an obstacle)
        if (new_row, new_col) not in self.obstacles:
            self.position = (new_row, new_col)

        # Increment step count
        self.step_count += 1

        # Check if we reached a goal
        if self.position in self.goal_pos:
            reward = self.terminal_reward
            done = True
        elif self.step_count >= self.max_steps:
            reward = 0.0
            done = True
        else:
            reward = -self.step_penalty
            done = False

        self._done = done

        # Get new state
        next_state = self.get_state()

        # Additional info
        info = {"position": self.position, "step_count": self.step_count}

        return next_state, reward, done, info

    def get_num_states(self):
        """
        Get the total number of states in the environment.

        Returns:
            int: Number of states
        """
        return self.width * self.height

    def get_num_actions(self):
        """
        Get the number of possible actions.

        Returns:
            int: Number of actions
        """
        return 4  # UP, RIGHT, DOWN, LEFT

    def render(self):
        """
        Render the grid world as a string.

        Returns:
            str: String representation of the grid
        """
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]

        # Mark obstacles
        for r, c in self.obstacles:
            if 0 <= r < self.height and 0 <= c < self.width:
                grid[r][c] = "#"

        # Mark goals
        for r, c in self.goal_pos:
            if 0 <= r < self.height and 0 <= c < self.width:
                grid[r][c] = "G"

        # Mark agent position
        r, c = self.position
        grid[r][c] = "A"

        # Convert to string
        result = ""
        for row in grid:
            result += " ".join(row) + "\n"

        return result

    def to_mdp(self):
        """
        Convert the grid world to a Markov Decision Process.

        Returns:
            MarkovDecisionProcess: MDP representation of the grid world
        """
        n_states = self.get_num_states()
        n_actions = self.get_num_actions()

        # Create transition and reward matrices
        transition_probs = np.zeros((n_states, n_actions, n_states))
        rewards = np.zeros((n_states, n_actions, n_states))

        # Terminal states
        terminal_states = [self.state_to_index(pos) for pos in self.goal_pos]

        # Fill in transition probabilities and rewards
        for row in range(self.height):
            for col in range(self.width):
                state = self.state_to_index((row, col))

                # Skip terminal states
                if state in terminal_states:
                    # Terminal state transitions to itself with reward 0
                    for a in range(n_actions):
                        transition_probs[state, a, state] = 1.0
                    continue

                # For each action
                for action in range(n_actions):
                    # Make a copy of the environment
                    env_copy = GridWorld(
                        width=self.width,
                        height=self.height,
                        start_pos=(row, col),
                        goal_pos=self.goal_pos,
                        obstacles=self.obstacles,
                        terminal_reward=self.terminal_reward,
                        step_penalty=self.step_penalty,
                    )

                    # Take the action
                    next_state, reward, _, _ = env_copy.step(action)

                    # Set transition probability to 1.0 for this state-action-next_state
                    transition_probs[state, action, next_state] = 1.0

                    # Set reward
                    rewards[state, action, next_state] = reward

        # Create the MDP
        mdp = MarkovDecisionProcess(
            n_states, n_actions, transition_probs, rewards, gamma=0.99, terminal_states=terminal_states
        )

        return mdp

    def state_to_index(self, position):
        """
        Convert position to state index.

        Args:
            position (tuple): (row, column)

        Returns:
            int: State index
        """
        row, col = position
        return row * self.width + col

    def index_to_state(self, index):
        """
        Convert state index to position.

        Args:
            index (int): State index

        Returns:
            tuple: (row, column)
        """
        row = index // self.width
        col = index % self.width
        return (row, col)

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space
        """
        return list(range(self.get_num_actions()))

    @property
    def state_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space
        """
        return list(range(self.get_num_states()))
