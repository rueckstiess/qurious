import numpy as np
import time
import os

from mini_rl.environments.grid_world import GridWorld
from mini_rl.policy import EpsilonGreedyPolicy, DeterministicTabularPolicy
from mini_rl.value_fns import TabularActionValueFunction
from mini_rl.agents import QLearningAgent, TabularAgent


def clear_screen():
    """Clear the console screen."""
    os.system("cls" if os.name == "nt" else "clear")


def train_agent(env, agent, num_episodes=10000, render=False, render_freq=100):
    """
    Train an agent in an environment.

    Args:
        env: Environment to train in
        agent: Agent to train
        num_episodes (int): Number of episodes to train for
        render (bool): Whether to render the environment
        render_freq (int): How often to render the environment

    Returns:
        list: Episode rewards
    """
    episode_rewards = []

    for episode in range(num_episodes):
        # Reset environment and agent
        state = env.reset()
        agent.reset()

        done = False
        episode_reward = 0

        # Render initial state if needed
        if render and episode % render_freq == 0:
            clear_screen()
            print(f"Episode {episode}")
            print(env.render())
            time.sleep(0.5)

        # Episode loop
        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Learn from experience
            agent.learn((state, action, reward, next_state, done))

            # Update state and accumulate reward
            state = next_state
            episode_reward += reward

            # Render if needed
            if render and episode % render_freq == 0:
                clear_screen()
                print(f"Episode {episode}, Step {env.step_count}, Reward {episode_reward}")
                print(env.render())
                time.sleep(0.5)

        # Record episode reward
        episode_rewards.append(episode_reward)

        # Print episode summary
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}: Average Reward = {avg_reward:.2f}")

    return episode_rewards


def visualize_policy(env, agent):
    """
    Visualize the agent's policy in the environment.

    Args:
        env: Environment
        agent: Agent with a policy
    """
    # Symbols for actions
    action_symbols = ["↑", "→", "↓", "←"]

    # Create grid of actions
    grid = [["" for _ in range(env.width)] for _ in range(env.height)]

    # Fill in actions
    for row in range(env.height):
        for col in range(env.width):
            state = env.state_to_index((row, col))

            # Skip obstacles
            if (row, col) in env.obstacles:
                grid[row][col] = "#"
                continue

            # For goals, just mark them
            if (row, col) in env.goal_pos:
                grid[row][col] = "G"
                continue

            # Get action from policy
            action = agent.choose_action(state)
            grid[row][col] = action_symbols[action]

    # Print the policy grid
    print("Learned Policy:")
    for row in grid:
        print(" ".join(row))


def run_evaluation(env, agent, num_episodes=10, render=True):
    """
    Evaluate an agent in an environment.

    Args:
        env: Environment to evaluate in
        agent: Agent to evaluate
        num_episodes (int): Number of episodes to evaluate for
        render (bool): Whether to render the environment

    Returns:
        float: Average reward
    """
    episode_rewards = []

    for episode in range(num_episodes):
        # Reset environment and agent
        state = env.reset()

        done = False
        episode_reward = 0

        # Render initial state if needed
        if render:
            clear_screen()
            print(f"Evaluation Episode {episode}")
            print(env.render())
            time.sleep(0.5)

        # Episode loop
        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Update state and accumulate reward
            state = next_state
            episode_reward += reward

            # Render if needed
            if render:
                clear_screen()
                print(f"Evaluation Episode {episode}, Step {env.step_count}, Reward {episode_reward}")
                print(env.render())
                time.sleep(0.5)

        # Record episode reward
        episode_rewards.append(episode_reward)

        print(f"Evaluation Episode {episode}: Reward = {episode_reward:.2f}")

    avg_reward = np.mean(episode_rewards)
    print(f"Average Evaluation Reward: {avg_reward:.2f}")

    return avg_reward


def main():
    """Main function to run the example."""
    # Create environment
    env = GridWorld(
        width=5,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 4)],
        obstacles=[(1, 1), (2, 1), (3, 1), (1, 3), (2, 3), (3, 3)],
        terminal_reward=10.0,
        step_penalty=0.1,
        max_steps=100,
    )

    print("Grid World Environment:")
    print(env.render())

    # Create agent components
    n_states = env.get_num_states()
    n_actions = env.get_num_actions()

    # Q-function
    q_function = TabularActionValueFunction(n_states, n_actions)

    # Base policy (will be updated based on Q-values)
    base_policy = DeterministicTabularPolicy(n_states, n_actions)

    # Epsilon-greedy exploration policy
    epsilon = 0.1
    policy = EpsilonGreedyPolicy(base_policy, epsilon)

    # Create Q-learning agent
    agent = QLearningAgent(policy, q_function, gamma=0.99)

    # Train the agent
    print("Training agent...")
    train_agent(env, agent, num_episodes=10000, render=False, render_freq=100)

    # Visualize the learned policy
    visualize_policy(env, agent)

    # # Evaluate the agent
    # print("\nEvaluating greedy agent...")
    # greedy_agent = QLearningAgent(base_policy, q_function, gamma=0.99)
    # greedy_agent = TabularAgent(base_policy)
    # run_evaluation(env, greedy_agent, num_episodes=3, render=True)


if __name__ == "__main__":
    main()
