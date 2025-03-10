import argparse
import random

from datasets import Dataset

from qurious.rl.agents import SarsaAgent
from qurious.rl.environments.grid_world import make_grid_world
from qurious.rl.utils import run_agent, train_agent
from qurious.visualization import AgentLayer, GridLayer, GridWorldVisualizer

GRIDWORLD_SYSTEM_PROMPT = """You are an expert in navigating grid world environments. You will be given a \
grid world environment and you need to find a path from the agent position to the goal position. \
The grid world is represented as a 2D ASCII representation, where . represents an empty cell, # represents an obstacle, A \
represents the agent and G represents the goal. You can move up, down, left, or right. Your task is to provide \
a sequence of comma-separated actions (up, down, left, right) that lead to the goal. Do not include any other text."""


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
    actions = " ".join([action_strs[a] for a in numeric_actions])

    print(env_ascii)
    if env.index_to_state(agent.experience.get_current_transition().next_state) not in env.goal_pos:
        print("goal not reached\n")
        return None
    print(f"Actions: {actions}\n")

    trajectory = {
        "env": env_ascii,
        "size": env.width,
        "actions": actions,
        "numeric_actions": numeric_actions,
        "n_steps": len(numeric_actions),
        "start_pos": env.start_pos,
        "goal_pos": env.goal_pos[0],
        "obstacles": env.obstacles,
    }

    return trajectory


def create_dataset(instances, chat_format=False, include_system_prompt=False):
    """Create a dataset from the given instances.
    Args:
        instances: List of instances

    Returns:
        DatasetDict
    """

    if chat_format:
        if include_system_prompt:
            instances = [
                {
                    "messages": [
                        {"role": "system", "content": GRIDWORLD_SYSTEM_PROMPT},
                        {"role": "user", "content": instance["env"]},
                        {"role": "assistant", "content": instance["actions"]},
                    ],
                    **instance,
                }
                for instance in instances
            ]
        else:
            instances = [
                {
                    "messages": [
                        {"role": "user", "content": instance["env"]},
                        {"role": "assistant", "content": instance["actions"]},
                    ],
                    **instance,
                }
                for instance in instances
            ]
    else:
        if include_system_prompt:
            instances = [
                {
                    "prompt": f"{GRIDWORLD_SYSTEM_PROMPT}\n\nGrid World:\n{instance['env']}\nActions:\n",
                    "response": instance["actions"],
                    **instance,
                }
                for instance in instances
            ]
        else:
            instances = [
                {
                    "prompt": f"Grid World:\n{instance['env']}\nActions:\n",
                    "response": instance["actions"],
                    **instance,
                }
                for instance in instances
            ]

    dataset = Dataset.from_list(instances)
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate grid world instances")
    parser.add_argument("--output", "-o", type=str, default="grid_world.jsonl", help="output file")
    parser.add_argument("--chat-format", "-c", action="store_true", help="use chat format")
    parser.add_argument("--num-instances", "-n", type=int, default=10, help="number of instances to generate")
    parser.add_argument("--min-grid-size", "-i", type=int, default=5, help="minimum grid size")
    parser.add_argument("--max-grid-size", "-a", type=int, default=10, help="maximum grid size")
    parser.add_argument("--include-system-prompt", "-s", action="store_true", help="include system prompt in dataset")
    args = parser.parse_args()

    all_trajectories = []

    while len(all_trajectories) < args.num_instances:
        print(f"Generating instance {len(all_trajectories) + 1} of {args.num_instances}\n\n")
        # Generate random grid size
        size = random.randint(args.min_grid_size, args.max_grid_size)
        env = make_grid_world(size=size)
        agent = SarsaAgent.from_env(env)

        train_agent(env, agent, num_episodes=2000)

        # switch to greedy policy
        agent.policy.epsilon = 0.0

        trajectory = collect_trajectory(env, agent)
        if trajectory:
            all_trajectories.append(trajectory)

    dataset = create_dataset(
        all_trajectories, chat_format=args.chat_format, include_system_prompt=args.include_system_prompt
    )

    # save trajectories as json file
    dataset.to_json(args.output)
    print(f"Saved {len(all_trajectories)} grid world instances to {args.output}")


if __name__ == "__main__":
    main()
