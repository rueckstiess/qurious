import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# import wandb
from datetime import datetime

from .utils import load_maze_data, evaluate_model
from .config import Config

from pathlib import Path

config = Config()
data_path = Path(config.data_dir)


def main():
    # Initialize wandb
    run_name = f"grid-world-sft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # wandb.init(
    #     project="qurious",
    #     name=run_name,
    #     config={
    #         "model": config.base_model,
    #         "learning_rate": config.sft_learning_rate,
    #         "epochs": config.sft_epochs,
    #         "batch_size": config.sft_batch_size,
    #         "lora_config": config.peft_config,
    #     },
    # )

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

    # Load conversation data (replace with your actual data)
    trajectories = load_maze_data(data_path / "trajectories_train.json")

    # Split data into training and evaluation sets
    train_data, eval_data = train_test_split(trajectories, test_size=0.2, random_state=42)

    # Create HF datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    def tokenize_chat(messages):
        return tokenizer.apply_chat_template(
            messages["messages"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            max_length=256,
            truncation=True,
            padding=True,
        )

    tokenized_train_dataset = train_dataset.map(tokenize_chat, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_chat, batched=True, remove_columns=eval_dataset.column_names)

    # print max length of tokenized sequences
    max_seq_length = max(len(x) for x in tokenized_train_dataset["input_ids"])
    print(f"Max length of tokenized sequences: {max_seq_length}")
    # wandb.log({"max_sequence_length": max_seq_length})

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments - optimized for Mac
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        per_device_eval_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=2,  # Increased to compensate for smaller batch size
        learning_rate=config.sft_learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
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
        # report_to="wandb",  # Enable wandb reporting
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=data_collator,
    )

    # Prepare test data (for this example, using eval_dataset)
    test_data = load_maze_data(data_path / "trajectories_test.json")

    # Evaluate before training
    accuracy, preds = evaluate_model(model, tokenizer, test_data, batch_size=16)
    print(f"Test Accuracy before training: {accuracy:.4f}")
    # wandb.log({"accuracy_before_training": accuracy})

    # Create a table to show example predictions before training
    # examples_table_before = wandb.Table(columns=["example_id", "maze", "prediction", "actual"])
    # for i in range(min(5, len(preds))):
    #     examples_table_before.add_data(i, test_data[i]["env"], preds[i], test_data[i]["actions"])
    # wandb.log({"examples_before_training": examples_table_before})

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")

    # Save the model
    output_dir = f"./grid_world_lora_adapter-{run_name}"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Evaluate
    accuracy, preds = evaluate_model(model, tokenizer, test_data, batch_size=16)
    print(f"Test Accuracy after training: {accuracy:.4f}")
    # wandb.log({"accuracy_after_training": accuracy})

    # Create a table to show example predictions after training
    # examples_table_after = wandb.Table(columns=["example_id", "maze", "prediction", "actual"])
    # for i in range(min(10, len(preds))):
    #     examples_table_after.add_data(i, test_data[i]["env"], preds[i], test_data[i]["actions"])
    # wandb.log({"examples_after_training": examples_table_after})

    # Print some examples
    for i in range(min(10, len(preds))):
        print(f"Example {i + 1}:")
        print(f"  Maze:\n{test_data[i]['env']}")
        print(f"  Predicted: {preds[i]}")
        print(f"  Actual: {test_data[i]['actions']}")
        print("")

    # Finish the wandb run
    # wandb.finish()


if __name__ == "__main__":
    main()
