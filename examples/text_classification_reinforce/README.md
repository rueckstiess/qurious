# End-to-End LLM REINFORCE Training Example

This example demonstrates how to train an LLM policy with the REINFORCE algorithm for a text classification task.

## Overview

This example implements:
1. A text classification environment that rewards correct classifications
2. An LLM-based policy specialized for classification tasks
3. A REINFORCE agent that updates the policy using policy gradients
4. A training script that puts it all together

## File Structure

- `text_environment.py` - Implements a text classification environment
- `classification_policy.py` - Specialized LLM policy for classification
- `train_reinforce.py` - Training script

The implementation makes use of:
- `qurious/rl/policies/llm_policy.py` - Base LLM policy implementation
- `qurious/rl/agents/policy_gradient.py` - REINFORCE agent implementation


## Running the Example

Train a model with default settings:
```
python train_reinforce.py
```

Customize training with command-line arguments:
```
python train_reinforce.py --model="EleutherAI/pythia-410m-deduped" --episodes=200 --lr=1e-6
```

Available arguments:
- `--model`: HuggingFace model name (default: "EleutherAI/pythia-410m-deduped")
- `--episodes`: Number of training episodes (default: 100)
- `--lr`: Learning rate (default: 5e-6)
- `--gamma`: Discount factor (default: 0.99)
- `--eval-interval`: How often to evaluate (default: 10)
- `--save-dir`: Directory to save models (default: "./models")

## Recommended Models

For quick experimentation:
- `EleutherAI/pythia-410m-deduped` (smaller, faster training)
- `microsoft/phi-1.5` (good balance of size and quality)

For better results:
- `meta-llama/Llama-2-7b-hf` (requires HF token)
- `mistralai/Mistral-7B-v0.1` (good performance, reasonable size)

## Expected Results

After training for 100 episodes:
- Initial accuracy: ~0.5 (random guessing)
- Final accuracy: 0.7-0.9 depending on model size

The training script saves:
- Checkpoints during training
- Final trained model
- Training curves plot

## Extending the Example

To adapt this to your own classification tasks:
1. Modify `create_sentiment_dataset()` to load your own data
2. Adjust the class labels in `ClassificationPolicy`
3. Tune learning rate and other hyperparameters

For more complex RL tasks:
1. Create a new environment with multi-step episodes
2. Consider using the `PPOAgent` for better sample efficiency