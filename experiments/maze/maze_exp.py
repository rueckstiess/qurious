import os

from datasets import load_dataset
from loguru import logger
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader

from qurious.experiments import BaseExperiment, Run
from qurious.llms.lora_manager import LoraManager
from qurious.llms.trainer import Trainer
from qurious.rl.environments.grid_world.utils import evaluate_model

# Env variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"


class FineTuneLLMsOnMazes(BaseExperiment):
    def prepare_dataset(self, dataset, tokenizer, system_prompt=None):
        """Prepare the dataset for training for chat and non-chat models."""

        # Create new "text" column by concatenating "prompt" and "response" columns
        def concat_prompt_response(example):
            if system_prompt:
                return {"text": system_prompt + "\n" + example["prompt"] + example["response"]}
            return {"text": example["prompt"] + example["response"]}

        def format_chat_messages(example):
            if system_prompt:
                # Add system prompt to the beginning of the messages
                example["messages"].insert(0, {"role": "system", "content": system_prompt})

            text = tokenizer.apply_chat_template(
                example["messages"],  # Exclude assistant message
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        if "messages" in dataset.column_names:
            # For chat-based models, format the messages
            dataset = dataset.map(format_chat_messages)
        else:
            # For non-chat models, use the prompts directly
            dataset = dataset.map(concat_prompt_response)
        return dataset

    def execute(self, run: Run):
        """Execute a training run."""

        config = run.config

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

        run.log_info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}%"
        )

        # Load and format the dataset
        dataset = load_dataset("json", data_files=config.data.file)["train"]
        dataset = self.prepare_dataset(dataset, tokenizer, config.data.system_prompt)

        # Split train and test sets
        dataset = dataset.train_test_split(test_size=0.2, seed=42)

        run.log_info(f"Training dataset contains {len(dataset['train'])} examples")
        run.log_info(f"Test dataset contains {len(dataset['test'])} examples")

        # Determine maximum length of input sequences
        MAX_LENGTH = max(len(tokenizer.encode(sample["text"])) for sample in dataset["train"]) + 10
        logger.info(f"Maximum length of input sequences: {MAX_LENGTH}")

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

        logger.info(
            f"Decoded input example:\n{tokenizer.decode(tokenized_train_dataset[0]['input_ids'], skip_special_tokens=False)}",
        )

        # Train Model

        # make data loaders for PyTorch format
        train_dataloader = DataLoader(
            tokenized_train_dataset.with_format("torch"), batch_size=config.training.batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(
            tokenized_eval_dataset.with_format("torch"), batch_size=config.training.batch_size, shuffle=False
        )

        optimizer = AdamW(peft_model.parameters(), lr=config.training.learning_rate, weight_decay=0.01)
        scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.1, total_iters=len(dataset["train"]))
        loss_fn = CrossEntropyLoss()

        trainer = Trainer(
            model=peft_model,
            config=config,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_fn=loss_fn,
            run=run,
        )

        # evaluate model before training
        accuracy, predictions = evaluate_model(
            peft_model, tokenizer, dataset["test"], batch_size=config.training.batch_size * 2
        )
        run.log_metrics({"accuracy": accuracy}, 0)

        result = trainer.train(
            train_dataloader=train_dataloader, eval_dataloader=eval_dataloader, num_epochs=config.training.epochs
        )
        run.log_info(f"Training result: {result}")

        # Loading best model
        run.log_info("Loading best model")
        peft_model = trainer.load_checkpoint("best_model", load_optimizer=False, load_scheduler=False)

        # evaluate model after training
        accuracy, predictions = evaluate_model(
            peft_model, tokenizer, dataset["test"], batch_size=config.training.batch_size * 2
        )
        run.log_metrics({"accuracy": accuracy}, trainer.step)

        # create table of predictions and references
        dataset["test"] = dataset["test"].add_column("predictions", predictions)
        df = dataset["test"].to_pandas()

        run.log_info(f"\n{df[['env', 'actions', 'predictions']]}")


if __name__ == "__main__":
    FineTuneLLMsOnMazes.main()
