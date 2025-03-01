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

from .utils import load_maze_data, evaluate_model
from .config import Config

from pathlib import Path

config = Config()
data_path = Path(config.data_dir)


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

    # Load conversation data (replace with your actual data)
    conversations = load_maze_data(data_path / "trajectories_train.json")

    # Split data into training and evaluation sets
    train_data, eval_data = train_test_split(conversations, test_size=0.2, random_state=42)

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
            max_length=200,
            truncation=True,
            padding=True,
        )

    tokenized_train_dataset = train_dataset.map(tokenize_chat, batched=True, remove_columns=["messages"])
    tokenized_eval_dataset = eval_dataset.map(tokenize_chat, batched=True, remove_columns=["messages"])

    # print max length of tokenized sequences
    print(f"Max length of tokenized sequences: {max(len(x) for x in tokenized_train_dataset['input_ids'])}")

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
    accuracy, preds, refs = evaluate_model(model, tokenizer, test_data, batch_size=16)
    print(f"Test Accuracy before training: {accuracy:.4f}")

    # Train the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model state...")

    # Save the model
    model.save_pretrained("./grid_world_lora_adapter")
    tokenizer.save_pretrained("./grid_world_lora_adapter")

    # Evaluate
    accuracy, preds, refs = evaluate_model(model, tokenizer, test_data, batch_size=16)
    print(f"Test Accuracy after training: {accuracy:.4f}")

    # Print some examples
    for i in range(min(10, len(preds))):
        print(f"Example {i + 1}:")
        print(f"  Maze:\n{test_data[i]['messages'][1]['content']}")
        print(f"  Predicted: {preds[i]}")
        print(f"  Actual: {refs[i]}")
        print("")


if __name__ == "__main__":
    main()
