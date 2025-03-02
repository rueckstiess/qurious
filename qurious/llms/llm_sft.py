import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

from .utils import evaluate_model, auto_device
from .config import Config

from pathlib import Path

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
        accuracy, preds = evaluate_model(
            self.model, self.tokenizer, self.test_data, max_samples=config.max_eval_samples, batch_size=self.batch_size
        )
        print(f"\nEvaluation - Step {state.global_step}, Accuracy: {accuracy:.4f}")

        # Add custom metrics to logs (will be combined with Trainer's metrics)
        if metrics is not None:
            metrics["accuracy"] = accuracy

        # Return the control object
        return control


def main():
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
    dataset = load_dataset("json", data_files=str(Path(config.data_dir) / "grid_world_1k.jsonl"))
    dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

    # print train and test sizes
    print(f"Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

    def tokenize_chat(messages):
        return tokenizer.apply_chat_template(
            messages["messages"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            max_length=300,
            truncation=True,
            padding=True,
        )

    tokenized_dataset = dataset.map(tokenize_chat, batched=True, remove_columns=dataset["train"].column_names)

    # print max length of tokenized sequences
    max_seq_length = max(len(x) for x in tokenized_dataset["train"]["input_ids"])
    print(f"Max length of tokenized sequences: {max_seq_length}")

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # mid-training evaluation on eval dataset
    custom_callback = CustomEvaluationCallback(
        model=model, tokenizer=tokenizer, eval_dataset=dataset["test"], batch_size=config.sft_batch_size
    )

    # Set up training arguments - optimized for Mac
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        per_device_eval_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=2,  # Increased to compensate for smaller batch size
        learning_rate=config.sft_learning_rate,
        warmup_steps=100,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=config.log_dir,
        logging_steps=config.log_interval,
        eval_strategy="steps",
        eval_steps=config.eval_interval,
        save_strategy="steps",
        save_steps=config.save_interval,
        bf16=False,  # Mac doesn't support bf16
        fp16=False,  # Mac Metal backend works best with defaults, not fp16
        optim="adamw_torch",  # Use standard PyTorch optimizer
        remove_unused_columns=False,
        report_to="none",  # Disable reporting to W&B
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        callbacks=[custom_callback],
    )

    # Evaluate before training
    accuracy, preds = evaluate_model(model, tokenizer, dataset["test"], batch_size=config.sft_batch_size)
    print(f"Test Accuracy before training: {accuracy:.4f}")

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")

    # Save the model
    output_dir = "./grid_world_lora_adapter"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    accuracy, preds = evaluate_model(model, tokenizer, dataset["test"], batch_size=config.sft_batch_size)
    print(f"Test Accuracy after training: {accuracy:.4f}")


if __name__ == "__main__":
    main()
