import random
import argparse
import json

from qurious.environments import GridWorld
from qurious.agents import SarsaAgent

from qurious.policy import DeterministicTabularPolicy, EpsilonGreedyPolicy
from qurious.value_fns import TabularActionValueFunction
from qurious.utils import train_agent, run_agent, clear_output

from qurious.visualization import GridWorldVisualizer, AgentLayer, GridLayer, PolicyLayer

from time import sleep


def make_env(width, height):
    # create random start position in grid
    start_pos = (random.randint(0, height - 1), random.randint(0, width - 1))

    # create random goal position in grid (different from start)
    while True:
        goal_pos = (random.randint(0, height - 1), random.randint(0, width - 1))
        if goal_pos != start_pos:
            break

    # Create a maze with a guaranteed path
    env = GridWorld(
        width=width,
        height=height,
        start_pos=start_pos,
        goal_pos=[goal_pos],
        obstacles=0.2,
        terminal_reward=0.0,
        step_penalty=0.1,
        max_steps=100,
    )
    return env


def create_agent(env):
    # Create agent components
    n_states = env.get_num_states()
    n_actions = env.get_num_actions()

    # Q-function
    q_function = TabularActionValueFunction(n_states, n_actions)

    # Base policy (will be updated based on Q-values)
    base_policy = DeterministicTabularPolicy(n_states, n_actions)

    # Epsilon-greedy exploration policy
    epsilon = 0.5
    policy = EpsilonGreedyPolicy(base_policy, epsilon, decay_rate=0.99)

    # Create agent
    agent = SarsaAgent(policy, q_function, gamma=0.99)
    agent.enable_experience_tracking()

    return agent


def visualize(env, agent):
    viz = GridWorldVisualizer(env)
    viz.add_layer(GridLayer())
    viz.add_layer(AgentLayer())
    viz.add_layer(PolicyLayer(agent.policy))

    def step_callback(*args, **kwargs):
        clear_output()
        print(viz.render_ascii())
        sleep(0.2)

    run_agent(env, agent, num_episodes=1, step_callback=step_callback)


def collect_trajectory(env, agent):
    agent.experience.clear()

    viz = GridWorldVisualizer(env)
    viz.add_layer(GridLayer())
    viz.add_layer(AgentLayer())

    # reset env and store first state
    env.reset()
    env_asciis = [viz.render_ascii()]
    print(env.render())

    def step_callback(*args, **kwargs):
        env_asciis.append(viz.render_ascii())

    run_agent(env, agent, num_episodes=1, step_callback=step_callback)

    action_strs = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    actions = [action_strs[transition.action] for transition in agent.experience]
    env_asciis = env_asciis[:-1]

    if env.index_to_state(agent.experience.get_current_transition().next_state) not in env.goal_pos:
        print("goal unreachable\n")
        trajectory = [{"env": env.render(), "action": "goal unreachable", "size": env.width}]
        return trajectory

    trajectory = [
        {"env": env_ascii, "action": action, "size": env.width} for env_ascii, action in zip(env_asciis, actions)
    ]
    return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate grid world trajectories")
    parser.add_argument("--num-trajectories", "-n", type=int, default=1, help="number of trajectories to generate")
    parser.add_argument("--min-grid-size", "-i", type=int, default=5, help="minimum grid size")
    parser.add_argument("--max-grid-size", "-a", type=int, default=10, help="maximum grid size")
    args = parser.parse_args()

    all_trajectories = []

    for _ in range(args.num_trajectories):
        # Generate random grid size
        size = random.randint(args.min_grid_size, args.max_grid_size)
        env = make_env(width=size, height=size)
        agent = create_agent(env)

        train_agent(env, agent, num_episodes=2000)

        # switch to greedy policy
        agent.policy = agent.policy.base_policy

        trajectory = collect_trajectory(env, agent)
        all_trajectories.extend(trajectory)

    # save trajectories as json file
    with open("trajectories.json", "w") as f:
        json.dump(all_trajectories, f)
    print("Saved trajectories to trajectories.json")


if __name__ == "__main__":
    main()
