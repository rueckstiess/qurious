import torch
import torch.nn as nn
from prepare_dataset import prepare_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from qurious.config import Config
from qurious.llms.lora_manager import LoraManager
from qurious.llms.trainer import Trainer


# Example 1: Training a simple PyTorch model
def train_simple_model():
    # Define a simple model
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))

    # Create a synthetic dataset
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # Create a basic config
    config = Config()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
    )

    # Train for 5 epochs
    history = trainer.train(
        train_dataloader=dataloader, num_epochs=5, save_dir="./checkpoints", early_stopping_patience=3
    )

    print(f"Training history: {history}")


# Example 2: Training an LLM with LoraManager
def train_lora_adapter():
    print("Training Lora adapter")

    # Create a LoraManager for the model
    config = Config(model={"base_model": "gpt2"})

    lora_manager = LoraManager(config)

    # Load gsm8k dataset
    train_dataloader, eval_dataloader = prepare_dataset(lora_manager.tokenizer)

    # Prepare trainer
    model = lora_manager.get_model("default")
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        config=config,
    )

    print("Before Training", trainer.evaluate(eval_dataloader))

    history = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=1,
        save_dir=None,
    )

    print(history)

    print("After Training", trainer.evaluate(eval_dataloader))


def train_base_model():
    print("Training base model")
    # Create a LoraManager for the model
    config = Config(model={"base_model": "gpt2"})

    lora_manager = LoraManager(config)

    # Load gsm8k dataset
    train_dataloader, eval_dataloader = prepare_dataset(lora_manager.tokenizer)

    # Prepare trainer
    model = lora_manager.get_base_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=config,
        scheduler=scheduler,
    )

    print("Before Training", trainer.evaluate(eval_dataloader))

    history = trainer.train(
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_epochs=1,
        save_dir=None,
    )

    print(history)

    print("After Training", trainer.evaluate(eval_dataloader))


# Example 3: Custom metrics and loss function
def train_with_custom_metrics():
    # Assuming you have a classification model
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 3),  # 3 classes
    )

    # Create a dataset for classification
    x = torch.randn(1000, 10)
    y = torch.randint(0, 3, (1000,))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Create a validation split
    val_x = torch.randn(200, 10)
    val_y = torch.randint(0, 3, (200,))
    val_dataset = TensorDataset(val_x, val_y)
    val_dataloader = DataLoader(val_dataset, batch_size=32)

    # Define optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    # Create a custom Trainer by subclassing
    class ClassificationTrainer(Trainer):
        def eval_step(self, batch):
            inputs, targets = self._prepare_batch(batch)
            outputs = self._forward_pass(inputs)
            loss = self._compute_loss(outputs, targets)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == targets).sum().item()
            total = targets.size(0)
            accuracy = correct / total

            return {"loss": loss.item(), "accuracy": accuracy}

        def evaluate(self, dataloader):
            self.model.eval()
            total_loss = 0.0
            total_accuracy = 0.0
            steps = 0

            with torch.no_grad():
                for batch in dataloader:
                    step_result = self.eval_step(batch)
                    total_loss += step_result["loss"]
                    total_accuracy += step_result["accuracy"]
                    steps += 1

            return {
                "loss": total_loss / steps if steps > 0 else 0,
                "accuracy": total_accuracy / steps if steps > 0 else 0,
            }

    # Initialize the custom trainer
    trainer = ClassificationTrainer(model=model, optimizer=optimizer, loss_fn=loss_fn)

    # Train with custom metrics
    history = trainer.train(
        train_dataloader=dataloader,
        eval_dataloader=val_dataloader,
        num_epochs=10,
        best_model_metric="eval_accuracy",  # Save based on accuracy, not loss
        save_dir="./classification_checkpoints",
    )

    print(f"Final validation accuracy: {history['eval_accuracy'][-1]:.4f}")


if __name__ == "__main__":
    train_base_model()
    train_lora_adapter()
