import numpy as np
import random
import matplotlib.pyplot as plt
from maze_environment import UP, DOWN, LEFT, RIGHT


class Agent:
    """Base class for agents that interact with the maze environment."""

    def __init__(self, env):
        """
        Initialize an agent with an environment.

        Args:
            env: MazeEnvironment instance
        """
        self.env = env

    def reset(self):
        """Reset the agent's internal state."""
        pass

    def get_action(self, state):
        """
        Get the next action based on the current state.

        Args:
            state: Current state (position in the maze)

        Returns:
            action: The next action to take
        """
        raise NotImplementedError("Subclasses must implement get_action()")

    def update(self, state, action, reward, next_state, done):
        """
        Update the agent's knowledge based on the transition.

        Args:
            state: Current state before action
            action: Action taken
            reward: Reward received
            next_state: New state after action
            done: Whether the episode is done
        """
        pass  # Default implementation does nothing


class RandomAgent(Agent):
    """An agent that selects random actions."""

    def get_action(self, state):
        """Choose a random action."""
        return random.choice([UP, DOWN, LEFT, RIGHT])


class ManualAgent(Agent):
    """An agent controlled by keyboard input."""

    def __init__(self, env, visualizer):
        """
        Initialize a manually-controlled agent.

        Args:
            env: MazeEnvironment instance
            visualizer: MazeVisualizer instance for display
        """
        super().__init__(env)
        self.visualizer = visualizer
        self.next_action = None
        self.running = False

        # Connect key event handler to the visualizer's figure
        self.key_event = self.visualizer.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

    def on_key_press(self, event):
        """Handle key press events for manual control."""
        if event.key == "up":
            self.next_action = UP
        elif event.key == "right":
            self.next_action = RIGHT
        elif event.key == "down":
            self.next_action = DOWN
        elif event.key == "left":
            self.next_action = LEFT
        elif event.key == "escape":
            self.running = False
        elif event.key == "r":
            self.env.reset()
            self.visualizer.update_display()

    def get_action(self, state):
        """
        Wait for keyboard input to determine the next action.

        Returns the action once received or None if no action is available.
        """
        return self.next_action

    def run_episode(self):
        """Run an interactive episode with manual control."""
        self.running = True
        done = False
        self.env.reset()
        self.visualizer.update_display()

        print("Use arrow keys to move. Press 'r' to reset, 'esc' to exit.")

        while self.running and not done:
            self.next_action = None

            # Wait for keyboard input
            while self.next_action is None and self.running:
                plt.pause(0.1)

            if not self.running:
                break

            # Take step in environment
            state, action = self.env.agent_pos, self.next_action
            next_state, reward, done = self.env.step(action)

            # Print current state and action
            print(self.env.path_history[-1])

            # Update visualization
            self.visualizer.update_display()

            if done:
                print("Goal reached! Episode complete.")

        return self.env.path_history


# class QlearningAgent(Agent):
#     """An agent that learns using Q-learning."""

#     def __init__(self, env, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.1):
#         """
#         Initialize a Q-learning agent.

#         Args:
#             env: MazeEnvironment instance
#             learning_rate: Alpha parameter for Q-learning
#             discount_factor: Gamma parameter for Q-learning
#             exploration_rate: Epsilon for epsilon-greedy exploration
#         """
#         super().__init__(env)
#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor
#         self.exploration_rate = exploration_rate

#         # Initialize Q-table with zeros
#         # State is represented as (row, col), and we have 4 actions per state
#         self.q_table = {}

#         # Initialize all state-action pairs to zero
#         for row in range(env.size):
#             for col in range(env.size):
#                 self.q_table[(row, col)] = {UP: 0.0, DOWN: 0.0, LEFT: 0.0, RIGHT: 0.0}

#     def get_action(self, state):
#         """
#         Choose an action using epsilon-greedy policy.

#         Args:
#             state: Current state (position in the maze)

#         Returns:
#             action: The selected action
#         """
#         # Exploration: choose a random action
#         if random.random() < self.exploration_rate:
#             return random.choice([UP, DOWN, LEFT, RIGHT])

#         # Exploitation: choose the best action from Q-table
#         return self._get_best_action(state)

#     def _get_best_action(self, state):
#         """
#         Get the action with the highest Q-value for the given state.

#         Args:
#             state: Current state

#         Returns:
#             action: The best action
#         """
#         q_values = self.q_table[state]
#         max_q = max(q_values.values())

#         # If multiple actions have the same max Q-value, choose randomly among them
#         best_actions = [action for action, q_value in q_values.items() if q_value == max_q]
#         return random.choice(best_actions)

#     def update(self, state, action, reward, next_state, done):
#         """
#         Update Q-values based on the observed transition.

#         Args:
#             state: Current state before action
#             action: Action taken
#             reward: Reward received
#             next_state: New state after action
#             done: Whether the episode is done
#         """
#         # Maximum Q-value for the next state
#         max_next_q = max(self.q_table[next_state].values()) if not done else 0

#         # Current Q-value
#         current_q = self.q_table[state][action]

#         # Q-learning update rule
#         new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

#         # Update Q-table
#         self.q_table[state][action] = new_q

#     def run_episode(self, max_steps=1000, visualizer=None):
#         """
#         Run a complete episode using the current policy.

#         Args:
#             max_steps: Maximum number of steps per episode
#             visualizer: Optional visualizer to display the environment

#         Returns:
#             path_history: History of positions, actions, and rewards
#         """
#         done = False
#         total_reward = 0
#         step_count = 0

#         # Reset environment
#         self.env.reset()
#         state = self.env.agent_pos

#         if visualizer:
#             visualizer.update_display()

#         while not done and step_count < max_steps:
#             # Choose action
#             action = self.get_action(state)

#             # Take action
#             next_state, reward, done = self.env.step(action)
#             total_reward += reward

#             # Update Q-values
#             self.update(state, action, reward, next_state, done)

#             # Move to next state
#             state = next_state
#             step_count += 1

#             if visualizer:
#                 visualizer.update_display()
#                 plt.pause(0.1)  # Short pause for visualization

#         return self.env.path_history, total_reward

#     def train(self, episodes=1000, max_steps=1000, visualize_every=None, visualizer=None):
#         """
#         Train the agent over multiple episodes.

#         Args:
#             episodes: Number of episodes to train
#             max_steps: Maximum steps per episode
#             visualize_every: If not None, visualize every nth episode
#             visualizer: Optional visualizer to display the environment

#         Returns:
#             rewards: List of total rewards per episode
#         """
#         rewards = []

#         for episode in range(episodes):
#             # Determine if this episode should be visualized
#             visualize = visualizer and visualize_every and episode % visualize_every == 0
#             vis = visualizer if visualize else None

#             # Run episode
#             _, episode_reward = self.run_episode(max_steps, vis)
#             rewards.append(episode_reward)

#             # Print progress
#             if episode % 100 == 0:
#                 print(f"Episode {episode}/{episodes}, Reward: {episode_reward}")

#         return rewards
