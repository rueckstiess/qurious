import os
from datetime import datetime

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer, apply_chat_template

from qurious.llms.config import Config
from qurious.llms.utils import (
    auto_device,
    evaluate_model,
    extract_actions_from_responses,
    load_dataset,
    run_actions_in_env,
)

config = Config()


class CustomEvaluationCallback(TrainerCallback):
    def __init__(self, model, tokenizer, eval_dataset, batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = eval_dataset
        self.batch_size = batch_size
        self.best_accuracy = 0.0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # This runs after the trainer's built-in evaluation
        accuracy, preds = evaluate_model(self.model, self.tokenizer, self.test_data, batch_size=self.batch_size)
        print(f"\nEvaluation - Step {state.global_step}, Accuracy: {accuracy:.4f}")

        # Add custom metrics to logs (will be combined with Trainer's metrics)
        if metrics is not None:
            metrics["accuracy"] = accuracy

        # Return the control object
        return control


def main():
    # run name and log dir
    run_name = f"grpo_run_{datetime.now().strftime('%Y%m%d_%H%M')}"
    log_dir = os.path.join("./logs", run_name)

    # Load model and tokenizer
    model_name = config.base_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure the tokenizer has a padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Set left padding for decoder-only models
    tokenizer.padding_side = "left"

    # Load model optimized for Mac Metal GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Define LoRA configuration - optimized for Mac
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        **config.peft_config,
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.to(auto_device())

    # Load train/eval data and split
    dataset = load_dataset("mongodb", db="gridworld", collection="gridworld_10k")
    dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
    dataset["eval"] = dataset["test"].shuffle(seed=42).select(range(config.max_eval_samples))

    # convert to prompt-only format for GRPO, see https://huggingface.co/docs/trl/main/en/dataset_formats#which-dataset-type-to-use
    prompt_dataset = dataset.map(lambda example: {"prompt": example["messages"][:-1]}, remove_columns=["messages"])
    prompt_dataset = prompt_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    # print train and test sizes
    print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['eval'])}, Test size: {len(dataset['test'])}")

    # mid-training evaluation on eval dataset
    custom_callback = CustomEvaluationCallback(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=dataset["eval"],
        batch_size=config.sft_batch_size,
    )

    # Reward functions
    def reward_goal_reached(completions, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            _, numeric_actions = extract_actions_from_responses(completion)
            example = {k: v[i] for k, v in kwargs.items()}
            solved = run_actions_in_env(example, numeric_actions)
            rewards.append(1.0 if solved else 0.0)
        return rewards

    def reward_num_steps(completions, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            _, numeric_actions = extract_actions_from_responses(completion)
            example = {k: v[i] for k, v in kwargs.items()}

            target_steps = example["n_steps"]
            actual_steps = len(numeric_actions)

            # Calculate difference as percentage of target steps
            if target_steps > 0:
                # Normalize difference to be between 0 and 1
                normalized_diff = min(abs(target_steps - actual_steps) / target_steps, 1.0)
                rewards.append(-normalized_diff)  # Make negative for penalty
            else:
                # Handle edge case where target_steps is 0
                rewards.append(-1.0 if actual_steps > 0 else 0.0)

        return rewards

    def reward_illegal_actions(completions, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            actions = [c.strip() for c in completion.split(",")]
            illegal_actions = [1 for a in actions if a not in ["up", "down", "left", "right"]]
            rewards.append(-len(illegal_actions) / len(actions))
        return rewards

    # Training arguments
    training_args = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=1,
        learning_rate=3e-4,
        warmup_steps=100,
        warmup_ratio=0.01,
        lr_scheduler_type="constant_with_warmup",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        max_prompt_length=512,
        max_completion_length=96,
        num_generations=8,
        optim="adamw_torch",
        bf16=False,
        report_to="tensorboard",
        remove_unused_columns=False,
        logging_dir=log_dir,
        logging_steps=1,
        label_names=[],
        eval_strategy="steps",
        eval_steps=50,
        reward_weights=[1.0, 0.3, 0.5],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_goal_reached, reward_num_steps, reward_illegal_actions],
        args=training_args,
        train_dataset=prompt_dataset["train"],
        eval_dataset=prompt_dataset["eval"],
        processing_class=tokenizer,
        callbacks=[custom_callback],
    )

    # trainer.processing_class.pad_token = tokenizer.eos_token

    # Evaluate before training (max_eval_samples samples)
    accuracy, preds = evaluate_model(model, tokenizer, dataset["eval"], batch_size=config.sft_batch_size)
    print(f"Test Accuracy before training: {accuracy:.4f}")

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")

    # Save the model
    output_dir = "./adapters/grid_world_grpo"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate after training (full test set)
    accuracy, preds = evaluate_model(model, tokenizer, dataset["test"], batch_size=config.sft_batch_size)
    print(f"Test Accuracy after training: {accuracy:.4f}")


if __name__ == "__main__":
    main()
