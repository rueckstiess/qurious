import argparse

import matplotlib.pyplot as plt
import torch
from classification_policy import ClassificationPolicy
from text_cls_env import TextClassificationEnvironment, create_sentiment_dataset
from tqdm import tqdm

from qurious.rl.agents.grpo import GRPOAgent


def train_with_grpo(
    model_name,
    num_episodes=100,
    lr=1e-6,
    gamma=0.99,
    clip_ratio=0.2,
    beta=0.04,
    group_size=16,
    process_supervision=True,
    eval_interval=10,
):
    """
    Train an LLM on text classification using GRPO.

    Args:
        model_name: Name of HuggingFace model to use
        num_episodes: Number of episodes to train for
        lr: Learning rate
        gamma: Discount factor
        clip_ratio: PPO clipping parameter
        beta: KL penalty coefficient
        group_size: Number of outputs to sample for each input
        process_supervision: Whether to use process supervision
        eval_interval: How often to evaluate
    """
    print("Creating dataset...")
    train_examples, val_examples = create_sentiment_dataset(n_examples=120)

    # Set up environment
    env = TextClassificationEnvironment(
        examples=train_examples, validation_examples=val_examples, classes=["positive", "negative"]
    )

    # Create policy
    print(f"Loading model {model_name}...")
    policy = ClassificationPolicy(
        model_name_or_path=model_name, device="cpu", model_kwargs={"torch_dtype": torch.float16}
    )
    ref_policy = ClassificationPolicy(
        model_name_or_path=model_name, device="cpu", model_kwargs={"torch_dtype": torch.float16}
    )

    # Create GRPO agent
    agent = GRPOAgent(
        policy=policy,
        reference_policy=ref_policy,
        gamma=gamma,
        lr=lr,
        clip_ratio=clip_ratio,
        beta=beta,
        group_size=group_size,
        process_supervision=process_supervision,
    )
    agent.enable_experience_tracking(enable_logging=True)

    # Training metrics
    rewards = []
    losses = []
    kl_divs = []
    accuracies = []

    # Initial evaluation
    print("Initial evaluation...")
    eval_acc = evaluate(env, agent)
    print(f"Initial accuracy: {eval_acc:.2f}")
    accuracies.append(eval_acc)

    # Training loop
    print(f"Training for {num_episodes} episodes with GRPO...")
    for episode in tqdm(range(num_episodes)):
        # Reset environment for new episode
        state = env.reset()
        agent.experience.clear()

        # Collect multiple outputs for the same input
        group_rewards = []

        # Sample group_size outputs for the current state
        for _ in range(group_size):
            # Reset environment to the same state
            env.reset(use_same_example=True)

            # Agent takes action (generate one complete response)
            action = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Record reward
            group_rewards.append(reward)

        # Learn from the group of experiences
        metrics = agent.learn(list(agent.experience.buffer))

        # Record metrics
        avg_reward = sum(group_rewards) / len(group_rewards)
        rewards.append(avg_reward)

        if metrics and "loss" in metrics:
            losses.append(metrics["loss"])
        if metrics and "kl_divergence" in metrics:
            kl_divs.append(metrics["kl_divergence"])

        # Print progress
        if (episode + 1) % 5 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Avg Reward: {avg_reward:.2f}, "
                f"Loss: {metrics.get('loss', 'N/A')}"
            )

        # Periodically evaluate
        if (episode + 1) % eval_interval == 0:
            eval_acc = evaluate(env, agent)
            accuracies.append(eval_acc)
            print(f"Episode {episode + 1}, Accuracy: {eval_acc:.4f}")

    # Final evaluation
    eval_acc = evaluate(env, agent)
    accuracies.append(eval_acc)
    print(f"Final accuracy: {eval_acc:.4f}")

    # Plot training curves
    plot_training_curves(rewards, losses, kl_divs, accuracies, eval_interval)

    return policy, agent, env


def evaluate(env, agent, num_episodes=20):
    """
    Evaluate the agent on the validation set.

    Args:
        env: Environment
        agent: Agent
        num_episodes: Number of episodes to evaluate on

    Returns:
        float: Accuracy
    """
    # Switch to validation mode
    env.set_validation_mode(True)

    correct = 0
    total = min(num_episodes, len(env.validation_examples))

    for _ in range(total):
        state = env.reset()
        action = agent.choose_action(state)
        _, _, _, info = env.step(action)
        if info["correct"]:
            correct += 1

    # Switch back to training mode
    env.set_validation_mode(False)

    return correct / total


def plot_training_curves(rewards, losses, kl_divs, accuracies, eval_interval):
    """
    Plot training curves.

    Args:
        rewards: List of episode rewards
        losses: List of training losses
        kl_divs: List of KL divergences
        accuracies: List of evaluation accuracies
        eval_interval: Evaluation interval
    """
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot losses
    if losses:
        ax2.plot(losses)
        ax2.set_title("GRPO Loss")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")

    # Plot KL divergence
    if kl_divs:
        ax3.plot(kl_divs)
        ax3.set_title("KL Divergence")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("KL")

    # Plot accuracies
    if accuracies:
        # Calculate x positions for accuracy points
        eval_episodes = [0] + [i * eval_interval for i in range(1, len(accuracies) - 1)] + [len(rewards)]
        ax4.plot(eval_episodes, accuracies, marker="o")
        ax4.set_title("Validation Accuracy")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Accuracy")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("grpo_training_curves.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM for text classification using GRPO")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m-deduped", help="HuggingFace model name")
    parser.add_argument("--episodes", type=int, default=50, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient")
    parser.add_argument("--group-size", type=int, default=16, help="Number of outputs to sample for each input")
    parser.add_argument("--process-supervision", action="store_true", help="Use process supervision")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")

    args = parser.parse_args()

    train_with_grpo(
        model_name=args.model,
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        clip_ratio=args.clip_ratio,
        beta=args.beta,
        group_size=args.group_size,
        process_supervision=args.process_supervision,
        eval_interval=args.eval_interval,
    )
