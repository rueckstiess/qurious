import random
import argparse
import json

from qurious.agents import SarsaAgent

from qurious.policy import DeterministicTabularPolicy, EpsilonGreedyPolicy
from qurious.value_fns import TabularActionValueFunction
from qurious.utils import train_agent, run_agent, clear_output

from qurious.visualization import GridWorldVisualizer, AgentLayer, GridLayer, PolicyLayer

from time import sleep

from .utils import make_env


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
    env_ascii = viz.render_ascii()

    run_agent(env, agent, num_episodes=1)

    action_strs = {
        0: "up",
        1: "right",
        2: "down",
        3: "left",
    }

    numeric_actions = [int(transition.action) for transition in agent.experience]
    actions = ", ".join([action_strs[a] for a in numeric_actions])

    print(env_ascii)
    if env.index_to_state(agent.experience.get_current_transition().next_state) not in env.goal_pos:
        print("goal not reached\n")
        return None
    print(f"Actions: {actions}\n")

    trajectory = [
        {
            "env": env_ascii,
            "size": env.width,
            "actions": actions,
            "numeric_actions": numeric_actions,
            "n_steps": len(numeric_actions),
            "start_pos": env.start_pos,
            "goal_pos": env.goal_pos[0],
            "obstacles": env.obstacles,
        }
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
        env = make_env(size=size)
        agent = create_agent(env)

        train_agent(env, agent, num_episodes=2000)

        # switch to greedy policy
        agent.policy = agent.policy.base_policy

        trajectory = collect_trajectory(env, agent)
        if trajectory:
            all_trajectories.append(trajectory)

    # save trajectories as json file
    with open("trajectories.json", "w") as f:
        json.dump(all_trajectories, f)
    print("Saved trajectories to trajectories.json")


if __name__ == "__main__":
    main()
