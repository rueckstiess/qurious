import argparse
import os
from pathlib import Path

import mlflow
import pandas as pd
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from qurious.config import Config, ConfigProduct
from qurious.llms.lora_manager import LoraManager
from qurious.llms.trainer import Trainer

# Env variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"


def main(config: Config, args: argparse.Namespace, parent_run_id: str = None):
    print(f"Running with config:\n{config.to_yaml()}")

    # create new "text" column by concatenating "prompt" and "response" columns
    def concat_prompt_response(example):
        return {"text": example["prompt"] + example["response"]}

    dataset = load_dataset("json", data_files=str(Path(config.paths.data_dir) / args.dataset))["train"]
    dataset = dataset.map(concat_prompt_response)

    # Split Train and Test Dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Training dataset contains {len(dataset['train'])} examples")
    print(f"Test dataset contains {len(dataset['test'])} examples")

    # Load models and tokenizer with LoraManager
    lora_manager = LoraManager(config)

    # Get the PEFT model
    peft_model = lora_manager.get_model("default")
    tokenizer = lora_manager.tokenizer

    # Make sure the model is in training mode and parameters require gradients
    peft_model.train()

    # Verify parameters require gradients
    trainable_params = 0
    all_param = 0
    for param in peft_model.parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    # Determine maximum length of input sequences
    MAX_LENGTH = max(len(tokenizer.encode(sample["text"])) for sample in dataset["train"]) + 1
    print(f"Maximum length of input sequences: {MAX_LENGTH}")

    def tokenize_and_pad_to_fixed_length(sample):
        result = tokenizer(
            sample["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train_dataset = dataset["train"].map(
        tokenize_and_pad_to_fixed_length, remove_columns=dataset["train"].column_names
    )
    tokenized_eval_dataset = dataset["test"].map(
        tokenize_and_pad_to_fixed_length, remove_columns=dataset["test"].column_names
    )

    assert all(len(x["input_ids"]) == MAX_LENGTH for x in tokenized_train_dataset)
    assert all(len(x["input_ids"]) == MAX_LENGTH for x in tokenized_eval_dataset)

    # assert that every attention_mask starts with a 0 (no example is cut off)
    assert all(x["attention_mask"][0] == 0 for x in tokenized_train_dataset if len(x["attention_mask"]) > 0)
    assert all(x["attention_mask"][0] == 0 for x in tokenized_eval_dataset if len(x["attention_mask"]) > 0)

    print(tokenizer.decode(tokenized_train_dataset[0]["input_ids"], skip_special_tokens=True))

    # Train Model

    # make data loaders for PyTorch format
    train_dataloader = DataLoader(tokenized_train_dataset.with_format("torch"), batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(tokenized_eval_dataset.with_format("torch"), batch_size=8, shuffle=False)

    optimizer = AdamW(peft_model.parameters(), lr=config.training.learning_rate, weight_decay=0.01)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=len(dataset["train"]))
    loss_fn = CrossEntropyLoss()

    trainer = Trainer(
        model=peft_model,
        config=config,
        optimizer=optimizer,
        scheduler=scheduler,
        loggers=["console" if args.debug else "mlflow"],
        experiment_name=args.experiment,
        loss_fn=loss_fn,
        parent_run_id=parent_run_id,
    )

    # Load checkpoint if provided
    if args.resume:
        checkpoint_path = Path(config.paths.checkpoint_dir) / args.resume
        trainer.load_checkpoint(checkpoint_path, load_optimizer=True, load_scheduler=True)
        print("Resuming from checkpoint:", checkpoint_path)

    if not args.skip_training:
        history = trainer.train(
            train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, num_epochs=config.training.epochs
        )
        print(f"Training history: {history}")

    # Loading best model
    best_model_path = str(Path(config.paths.checkpoint_dir) / "best_model.pt")
    trainer.load_checkpoint(best_model_path, load_optimizer=False, load_scheduler=False)

    # ## Generate outputs

    eval_samples = dataset["test"].select(range(10))
    # model = lora_manager.get_base_model()
    # model = peft_model
    model = trainer.model

    # Generate outputs
    results = []
    for sample in tqdm(eval_samples):
        # Assuming your dataset has 'input' and 'target' fields
        # Adjust the field names as needed for your specific dataset
        input_text = sample["prompt"]
        reference = sample["response"]

        # Tokenize and generate
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # Adjust as needed
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the newly generated part (optional)
        # This is model and tokenizer specific, you may need to adjust
        generated_response = generated_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :]

        results.append({"input": input_text, "generated": generated_response, "reference": reference})

    df = pd.DataFrame(results)
    print(df)

    if not args.debug:
        with mlflow.start_run(trainer.run_id):
            mlflow.log_artifact(best_model_path, artifact_path="best_model.pt")
            mlflow.log_table(df, artifact_file="generated_samples.json")
    else:
        print("Generated samples:\n")
        for result in results:
            print(f"Input: {result['input']}")
            print(f"Generated: {result['generated']}")
            print(f"Reference: {result['reference']}")
            print("-" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset.")
    parser.add_argument("--config", "-c", type=str, default="./config.yaml", help="Path to the config file.")
    parser.add_argument("--dataset", "-d", type=str, default="grid_world_1k.jsonl", help="Dataset filename.")
    parser.add_argument("--experiment", "-e", type=str, default="maze-supervised-finetune", help="Experiment name.")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Path to the checkpoint to resume from.")
    parser.add_argument("--skip-training", action="store_true", help="Skip training and only generate outputs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (log to console instead of MLflow).")
    parser.add_argument("--params", "-p", nargs="+", default=[], help="Parameters to override in the config file.")

    args = parser.parse_args()

    config = Config.from_yaml_file(args.config)

    if args.params:
        override_args = ["--params", *args.params]
        override_config = Config.from_args(override_args)
        config = config.merge(override_config)

    multi_config = ConfigProduct(config)
    if len(multi_config) > 1:
        # create top-level run in mlflow and pass parent_run_id to child runs
        mlflow.set_experiment(args.experiment)
        mlflow.start_run()
        mlflow.log_params(config.to_dict())
        print(f"Found {len(multi_config)} configurations. Running all of them.")
        for config in multi_config:
            main(config=config, args=args, parent_run_id=mlflow.active_run().info.run_id)
        mlflow.end_run()

    else:
        main(config=config, args=args)
