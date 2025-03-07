"""
Configuration file for the LLM SQL Explorer project.
"""


class Config:
    # Model Settings
    base_model = "meta-llama/Llama-3.2-3B-Instruct"  # Base LLM
    peft_config = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",  # Reduced parameter count
        "bias": "none",
    }

    # SFT Settings
    sft_epochs = 1
    sft_batch_size = 4  # Batch size for SFT and eval
    sft_learning_rate = 1e-4  # SFT learning rate

    # Logging and Evaluation
    log_interval = 10  # Log every N episodes
    eval_interval = 100  # Evaluate every N episodes
    save_interval = 1000  # Save models every N episodes
    max_eval_samples = 50  # Maximum samples for evaluation

    # Paths
    output_dir = "./outputs"  # Output directory for logs and models
    log_dir = "./logs"  # Log directory
    data_dir = "./data"  # Data directory

    # # Environment Settings
    # max_query_length = 1024  # Maximum tokens in SQL query
    # max_context_length = 2048  # Maximum tokens in context window

    # # Data Collection
    # num_episodes = 100  # Number of episodes to collect
    # max_steps_per_episode = 20  # Maximum steps per episode

    # # PPO Settings
    # ppo_epochs = 4  # Number of PPO epochs
    # batch_size = 16  # Batch size for PPO updates
    # mini_batch_size = 4  # Mini-batch size for PPO updates
    # gamma = 0.99  # Discount factor
    # gae_lambda = 0.95  # GAE lambda parameter
    # clip_range = 0.2  # PPO clip range
    # value_coef = 0.5  # Value loss coefficient
    # kl_coef = 0.1  # KL divergence coefficient
    # entropy_coef = 0.01  # Entropy coefficient
    # max_grad_norm = 0.5  # Gradient norm clipping
    # ppo_learning_rate = 1e-5  # PPO learning rate

    # # Surprise Calculation
    # reward_scale = 1.0  # Scaling factor for rewards
    # reward_shift = 0.0  # Shift for rewards

    # # Tracking
    # use_wandb = False  # Whether to use Weights & Biases for tracking
    # wandb_project = "llm-explorer"  # W&B project name
