import matplotlib.pyplot as plt
import numpy as np

from qurious.rl.environments import GridWorld
from qurious.rl.experience import Transition
from qurious.rl.policies import DeterministicTabularPolicy, EpsilonGreedyPolicy, SoftmaxPolicy
from qurious.rl.value_fns import TabularActionValueFunction, TabularStateValueFunction
from qurious.visualization import (
    AgentLayer,
    EligibilityTraceLayer,
    EpisodeAnimator,
    GridLayer,
    GridWorldVisualizer,
    PolicyLayer,
    StateValueLayer,
    TrajectoryLayer,
    create_gridworld_visualizer,
)


def example_basic_visualization():
    """Example of basic grid world visualization."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a visualizer with just the grid and agent
    vis = GridWorldVisualizer(env)
    vis.add_layer(GridLayer())
    vis.add_layer(AgentLayer())

    # Render as ASCII
    print("ASCII Rendering:")
    print(vis.render_ascii())

    # Render with matplotlib
    print("\nRendering with matplotlib...")
    fig, ax = vis.render_matplotlib()
    plt.savefig("grid_world_basic.png", dpi=150)
    plt.close(fig)
    print("Saved to 'grid_world_basic.png'")


def example_policy_visualization():
    """Example of policy visualization."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a deterministic policy
    n_states = env.get_n_states()
    n_actions = env.get_n_actions()
    det_policy = DeterministicTabularPolicy(n_states, n_actions)

    # Set some specific actions for the policy
    for s in range(n_states):
        # Convert state index to row, col
        row, col = env.index_to_state(s)

        # Simple policy: move right and down towards the goal
        if row < 4:  # If not at the bottom row
            det_policy.update(s, GridWorld.DOWN)
        else:  # Bottom row, move right
            det_policy.update(s, GridWorld.RIGHT)

    # Create an epsilon-greedy policy based on the deterministic policy
    eps_policy = EpsilonGreedyPolicy(det_policy, epsilon=0.2)

    # Create a softmax policy with some random values
    soft_policy = SoftmaxPolicy(n_states, n_actions, temperature=0.5)
    for s in range(n_states):
        # Set some random Q values
        action_values = np.random.randn(n_actions)
        for a, val in enumerate(action_values):
            soft_policy.update(s, a, val)

    # Visualize the deterministic policy
    vis1 = create_gridworld_visualizer(env, policy=det_policy)
    vis1.render_matplotlib()[0].savefig("grid_world_det_policy.png", dpi=150)

    # Visualize the epsilon-greedy policy
    vis2 = create_gridworld_visualizer(env, policy=eps_policy)
    vis2.render_matplotlib()[0].savefig("grid_world_eps_policy.png", dpi=150)

    # Visualize the softmax policy
    vis3 = create_gridworld_visualizer(env, policy=soft_policy)
    vis3.render_matplotlib()[0].savefig("grid_world_soft_policy.png", dpi=150)

    print(
        "Policy visualizations saved to 'grid_world_det_policy.png', 'grid_world_eps_policy.png', and 'grid_world_soft_policy.png'"
    )


def example_value_function_visualization():
    """Example of value function visualization."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a state value function
    n_states = env.get_n_states()
    n_actions = env.get_n_actions()
    v_func = TabularStateValueFunction(n_states, initial_value=0.0)

    # Set some state values (distance-based heuristic)
    goal_pos = env.goal_pos[0]
    for s in range(n_states):
        pos = env.index_to_state(s)
        # Manhattan distance to goal
        distance = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
        # Value decreases with distance
        value = 10.0 / (1.0 + distance)
        v_func.values[s] = value

    # Create an action value function
    q_func = TabularActionValueFunction(n_states, n_actions, initial_value=0.0)

    # Set some Q-values
    for s in range(n_states):
        pos = env.index_to_state(s)
        row, col = pos

        # For each action
        for a in range(n_actions):
            # Calculate next position after the action
            if a == GridWorld.UP and row > 0:
                next_row, next_col = row - 1, col
            elif a == GridWorld.RIGHT and col < env.width - 1:
                next_row, next_col = row, col + 1
            elif a == GridWorld.DOWN and row < env.height - 1:
                next_row, next_col = row + 1, col
            elif a == GridWorld.LEFT and col > 0:
                next_row, next_col = row, col - 1
            else:
                next_row, next_col = row, col  # Stay in place

            # Check if it's an obstacle
            if (next_row, next_col) in env.obstacles:
                next_row, next_col = row, col  # Stay in place

            # Manhattan distance from next position to goal
            distance = abs(next_row - goal_pos[0]) + abs(next_col - goal_pos[1])
            # Q-value decreases with distance
            q_value = 10.0 / (1.0 + distance)
            q_func.values[s, a] = q_value

    # Visualize the state value function
    vis1 = create_gridworld_visualizer(env, value_fn=v_func)
    vis1.render_matplotlib()[0].savefig("grid_world_state_value.png", dpi=150)

    # Visualize the action value function
    vis2 = create_gridworld_visualizer(env, action_value_fn=q_func)
    vis2.get_layer("ActionValue").enabled = True
    vis2.render_matplotlib()[0].savefig("grid_world_action_value.png", dpi=150)

    # Visualize both policy and value function
    det_policy = DeterministicTabularPolicy(n_states, n_actions)
    for s in range(n_states):
        # Choose the action with highest Q-value
        best_action = q_func.get_best_action(s)
        det_policy.update(s, best_action)

    vis3 = create_gridworld_visualizer(env, policy=det_policy, value_fn=v_func)
    vis3.render_matplotlib()[0].savefig("grid_world_policy_and_value.png", dpi=150)

    print(
        "Value function visualizations saved to 'grid_world_state_value.png', 'grid_world_action_value.png', and 'grid_world_policy_and_value.png'"
    )


def example_trajectory_visualization():
    """Example of trajectory visualization."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a list of transitions representing a path
    transitions = []

    # Define a path to the goal
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]

    # Create transitions from the path
    for i in range(len(path) - 1):
        state = env.state_to_index(path[i])
        next_state = env.state_to_index(path[i + 1])

        # Determine action based on the direction of movement
        if path[i + 1][0] > path[i][0]:  # Moving down
            action = GridWorld.DOWN
        elif path[i + 1][0] < path[i][0]:  # Moving up
            action = GridWorld.UP
        elif path[i + 1][1] > path[i][1]:  # Moving right
            action = GridWorld.RIGHT
        else:  # Moving left
            action = GridWorld.LEFT

        # Reward is 10 for reaching the goal, -0.1 otherwise
        reward = 10.0 if i == len(path) - 2 else -0.1
        done = i == len(path) - 2

        transitions.append(Transition(state, action, reward, next_state, done))

    # Create a visualizer with the trajectory
    vis = create_gridworld_visualizer(env)

    # Add the trajectory layer
    traj_layer = TrajectoryLayer(transitions, show_rewards=True)
    vis.add_layer(traj_layer)

    # Render the trajectory
    vis.render_matplotlib()[0].savefig("grid_world_trajectory.png", dpi=150)
    print("Trajectory visualization saved to 'grid_world_trajectory.png'")

    # Create an animation
    animator = EpisodeAnimator(env, vis, transitions, interval=500)
    animator.animate(save_path="grid_world_animation.gif", fps=2, dpi=100)
    print("Animation saved to 'grid_world_animation.gif'")


def example_eligibility_trace():
    """Example of eligibility trace visualization."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a state value function
    n_states = env.get_n_states()
    TabularStateValueFunction(n_states, initial_value=0.0)

    # Create mock eligibility traces
    traces = np.zeros(n_states)

    # Define a path with decreasing eligibility
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]

    # Set eligibility values along the path (decreasing)
    for i, pos in enumerate(path):
        state = env.state_to_index(pos)
        traces[state] = 1.0 * (0.9**i)  # Exponential decay

    # Create a visualizer with the eligibility trace
    vis = create_gridworld_visualizer(env)

    # Add the eligibility trace layer
    trace_layer = EligibilityTraceLayer(traces, cmap="hot", alpha=0.7)
    vis.add_layer(trace_layer)

    # Render the eligibility trace
    vis.render_matplotlib()[0].savefig("grid_world_eligibility.png", dpi=150)
    print("Eligibility trace visualization saved to 'grid_world_eligibility.png'")


def example_advanced_composition():
    """Example of composing multiple visualization elements."""
    # Create a simple grid world
    env = GridWorld(
        width=6,
        height=5,
        start_pos=(0, 0),
        goal_pos=[(4, 5)],
        obstacles=[(1, 1), (1, 2), (2, 3), (3, 1)],
        terminal_reward=10.0,
        step_penalty=0.1,
    )

    # Create a state value function
    n_states = env.get_n_states()
    n_actions = env.get_n_actions()
    v_func = TabularStateValueFunction(n_states, initial_value=0.0)

    # Set some state values (distance-based heuristic)
    goal_pos = env.goal_pos[0]
    for s in range(n_states):
        pos = env.index_to_state(s)
        # Manhattan distance to goal
        distance = abs(pos[0] - goal_pos[0]) + abs(pos[1] - goal_pos[1])
        # Value decreases with distance
        value = 10.0 / (1.0 + distance)
        v_func.values[s] = value

    # Create a policy
    policy = DeterministicTabularPolicy(n_states, n_actions)

    # Define a policy that moves towards the goal
    for s in range(n_states):
        pos = env.index_to_state(s)
        row, col = pos

        # Simple policy: if closer to goal by moving right, do that; otherwise move down
        if col < goal_pos[1]:
            policy.update(s, GridWorld.RIGHT)
        elif row < goal_pos[0]:
            policy.update(s, GridWorld.DOWN)
        else:
            # Default to right
            policy.update(s, GridWorld.RIGHT)

    # Create a trail of transitions
    transitions = []

    # Define a path to the goal
    path = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]

    # Create transitions from the path
    for i in range(len(path) - 1):
        state = env.state_to_index(path[i])
        next_state = env.state_to_index(path[i + 1])

        # Determine action based on the direction of movement
        if path[i + 1][0] > path[i][0]:  # Moving down
            action = GridWorld.DOWN
        elif path[i + 1][0] < path[i][0]:  # Moving up
            action = GridWorld.UP
        elif path[i + 1][1] > path[i][1]:  # Moving right
            action = GridWorld.RIGHT
        else:  # Moving left
            action = GridWorld.LEFT

        # Reward is 10 for reaching the goal, -0.1 otherwise
        reward = 10.0 if i == len(path) - 2 else -0.1
        done = i == len(path) - 2

        transitions.append(Transition(state, action, reward, next_state, done))

    # Create mock eligibility traces
    traces = np.zeros(n_states)

    # Set eligibility values along the path (decreasing)
    for i, pos in enumerate(path):
        state = env.state_to_index(pos)
        traces[state] = 1.0 * (0.9**i)  # Exponential decay

    # Create a visualizer with multiple layers
    vis = GridWorldVisualizer(env)
    vis.add_layer(GridLayer())
    vis.add_layer(AgentLayer())
    vis.add_layer(StateValueLayer(v_func, alpha=0.3))
    vis.add_layer(PolicyLayer(policy, arrow_scale=0.8))
    vis.add_layer(TrajectoryLayer(transitions))
    vis.add_layer(EligibilityTraceLayer(traces, alpha=0.3))

    # Customize the visualizer
    vis.configure(
        color_empty="white", color_obstacle="slategray", color_goal="lime", color_agent="red", figsize=(10, 8), dpi=150
    )

    # Render the combined visualization
    vis.render_matplotlib()[0].savefig("grid_world_combined.png", dpi=150)
    print("Combined visualization saved to 'grid_world_combined.png'")


if __name__ == "__main__":
    print("Running visualization examples...")

    example_basic_visualization()
    example_policy_visualization()
    example_value_function_visualization()
    example_trajectory_visualization()
    example_eligibility_trace()
    example_advanced_composition()

    print("All examples completed!")
