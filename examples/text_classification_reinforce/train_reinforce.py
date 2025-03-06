import argparse
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from classification_policy import ClassificationPolicy
from text_cls_env import TextClassificationEnvironment, create_sentiment_dataset

from qurious.rl.agents.policy_gradient import REINFORCEAgent


def train(model_name, num_episodes=100, lr=5e-6, gamma=0.99, eval_interval=10, save_dir="./models"):
    """
    Train an LLM on text classification using REINFORCE.

    Args:
        model_name: Name of HuggingFace model to use
        num_episodes: Number of episodes to train for
        lr: Learning rate
        gamma: Discount factor
        eval_interval: How often to evaluate
        save_dir: Directory to save model
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
        model_name_or_path=model_name, classes=env.classes, model_kwargs={"torch_dtype": torch.float16}
    )

    # Create agent
    agent = REINFORCEAgent(policy=policy, gamma=gamma, lr=lr)
    agent.enable_experience_tracking()

    # Training metrics
    rewards = []
    losses = []
    accuracies = []

    # Initial evaluation
    print("Initial evaluation...")
    eval_acc = evaluate(env, agent)
    print(f"Initial accuracy: {eval_acc:.2f}")
    accuracies.append(eval_acc)

    # Set up save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"classification_reinforce_{timestamp}")
    os.makedirs(save_path, exist_ok=True)

    # Training loop
    print(f"Training for {num_episodes} episodes...")
    start_time = time.time()

    for episode in range(num_episodes):
        # Reset environment and agent for new episode
        state = env.reset()
        done = False
        episode_reward = 0

        # Agent takes one step (classification tasks are one-step)
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        # Store experience and update policy
        agent.store_experience(state, action, reward, next_state, done)

        # Only learn when episode is complete
        if done:
            metrics = agent.learn(agent.experience.get_current_episode())
            episode_reward = reward
            rewards.append(episode_reward)

            if metrics and "loss" in metrics:
                losses.append(metrics["loss"])

            # Print progress
            print(
                f"Episode {episode + 1}/{num_episodes}, "
                f"Reward: {episode_reward:.2f}, "
                f"Prediction: {info['predicted']}, "
                f"True: {info['true_label']}"
            )

        # Periodically evaluate
        if (episode + 1) % eval_interval == 0:
            eval_acc = evaluate(env, agent)
            accuracies.append(eval_acc)
            print(f"Episode {episode + 1}, Accuracy: {eval_acc:.4f}")

            # Save model checkpoint
            checkpoint_path = os.path.join(save_path, f"checkpoint_ep{episode + 1}")
            policy.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Final evaluation
    eval_acc = evaluate(env, agent)
    accuracies.append(eval_acc)
    print(f"Final accuracy: {eval_acc:.4f}")

    # Calculate training time
    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f} seconds")

    # Save final model
    final_path = os.path.join(save_path, "final_model")
    policy.save(final_path)
    print(f"Saved final model to {final_path}")

    # Plot training curves
    plot_training_curves(rewards, losses, accuracies, save_path)

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


def plot_training_curves(rewards, losses, accuracies, save_path):
    """
    Plot and save training curves.

    Args:
        rewards: List of episode rewards
        losses: List of training losses
        accuracies: List of evaluation accuracies
        save_path: Directory to save plots
    """
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Plot rewards
    ax1.plot(rewards)
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot losses if available
    if losses:
        ax2.plot(losses)
        ax2.set_title("Training Loss")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")

    # Plot accuracies
    eval_episodes = np.linspace(0, len(rewards), len(accuracies))
    ax3.plot(eval_episodes, accuracies)
    ax3.set_title("Validation Accuracy")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Accuracy")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an LLM for text classification using REINFORCE")
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-410m-deduped", help="HuggingFace model name")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--eval-interval", type=int, default=10, help="Evaluation interval")
    parser.add_argument("--save-dir", type=str, default="./models", help="Directory to save models")

    args = parser.parse_args()

    train(
        model_name=args.model,
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
    )
