import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qurious.config import Config
from qurious.llms.trainer import Trainer


class SimpleModel(nn.Module):
    """Simple model for testing the Trainer class."""

    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class ModelWithLoss(nn.Module):
    """Model that returns a loss in its output, similar to HuggingFace models."""

    def __init__(self, input_dim=10, hidden_dim=5, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, targets=None):
        x = self.relu(self.fc1(x))
        outputs = self.fc2(x)

        # Create a result object with a loss attribute
        class Output:
            pass

        result = Output()
        result.logits = outputs

        if targets is not None:
            # Simple MSE loss
            result.loss = torch.mean((outputs - targets) ** 2)
        else:
            result.loss = None

        return result


@pytest.fixture
def model():
    """Fixture for a simple model."""
    return SimpleModel(input_dim=10, hidden_dim=5, output_dim=1)


@pytest.fixture
def model_with_loss():
    """Fixture for a model that returns loss."""
    return ModelWithLoss(input_dim=10, hidden_dim=5, output_dim=1)


@pytest.fixture
def optimizer(model):
    """Fixture for an optimizer."""
    return optim.SGD(model.parameters(), lr=0.01)


@pytest.fixture
def loss_fn():
    """Fixture for a loss function."""
    return nn.MSELoss()


@pytest.fixture
def config():
    """Fixture for training configuration."""
    # Create a config with the structure based on the new Config class
    config = Config(
        {
            "training": {
                "learning_rate": 0.01,
                "max_grad_norm": 1.0,
                "scheduler_step_per_batch": True,
                "log_interval": 10,
                "save_interval": 1,
                "checkpoint_dir": "./checkpoints",
            },
        }
    )
    return config


@pytest.fixture
def dataloader():
    """Fixture for a simple dataloader."""
    # Create random data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=16)


class TestTrainer:
    def test_initialization(self, model, loss_fn, config):
        """Test trainer initialization."""
        trainer = Trainer(model, loss_fn, config=config)

        assert trainer.model == model
        assert trainer.loss_fn == loss_fn
        assert trainer.config == config
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.scheduler is None

        # Test device assignment
        assert isinstance(trainer.device, torch.device)

    def test_custom_optimizer(self, model, loss_fn, optimizer, config):
        """Test trainer with custom optimizer."""
        trainer = Trainer(model, loss_fn, optimizer=optimizer, config=config)

        assert trainer.optimizer == optimizer

    def test_get_default_optimizer(self, model, loss_fn, config):
        """Test the default optimizer creation."""
        trainer = Trainer(model, loss_fn, config=config)

        default_optimizer = trainer._get_default_optimizer(0.001)
        assert isinstance(default_optimizer, torch.optim.AdamW)
        assert default_optimizer.param_groups[0]["lr"] == 0.001

    def test_prepare_batch_tensor(self, model, loss_fn, config):
        """Test preparing a tensor batch."""
        trainer = Trainer(model, loss_fn, config=config)

        # Test with a single tensor
        batch = torch.randn(10, 10)
        inputs, targets = trainer._prepare_batch(batch)

        assert str(inputs.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert targets is None

    def test_prepare_batch_tuple(self, model, loss_fn, config):
        """Test preparing a tuple batch."""
        trainer = Trainer(model, loss_fn, config=config)

        # Test with a tuple of tensors
        inputs_raw = torch.randn(10, 10)
        targets_raw = torch.randn(10, 1)
        batch = (inputs_raw, targets_raw)

        inputs, targets = trainer._prepare_batch(batch)

        assert str(inputs.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert str(targets.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert torch.allclose(inputs.cpu(), inputs_raw)
        assert torch.allclose(targets.cpu(), targets_raw)

    def test_prepare_batch_dict(self, model, loss_fn, config):
        """Test preparing a dict batch."""
        trainer = Trainer(model, loss_fn, config=config)

        # Test with a dictionary
        batch = {"input_ids": torch.randint(0, 100, (5, 10)), "attention_mask": torch.ones(5, 10)}

        inputs, targets = trainer._prepare_batch(batch)

        assert isinstance(inputs, dict)
        assert targets is None
        assert str(inputs["input_ids"].device).split(":")[0] == str(trainer.device).split(":")[0]
        assert str(inputs["attention_mask"].device).split(":")[0] == str(trainer.device).split(":")[0]

    def test_forward_pass_standard(self, model, loss_fn, config):
        """Test forward pass with a standard model."""
        trainer = Trainer(model, loss_fn, config=config)

        inputs = torch.randn(10, 10).to(trainer.device)
        outputs = trainer._forward_pass(inputs)

        assert outputs.shape == (10, 1)

    def test_forward_pass_dict(self, model_with_loss, loss_fn, config):
        """Test forward pass with dict inputs (HuggingFace-style)."""
        trainer = Trainer(model_with_loss, loss_fn, config=config)

        inputs = {"x": torch.randn(10, 10).to(trainer.device), "targets": torch.randn(10, 1).to(trainer.device)}

        outputs = trainer._forward_pass(inputs)

        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_compute_loss_standard(self, model, loss_fn, config):
        """Test loss computation with standard model."""
        trainer = Trainer(model, loss_fn, config=config)

        outputs = torch.randn(10, 1).to(trainer.device)
        targets = torch.randn(10, 1).to(trainer.device)

        loss = trainer._compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar tensor

    def test_compute_loss_with_model_loss(self, model_with_loss, loss_fn, config):
        """Test loss computation with a model that returns loss."""
        trainer = Trainer(model_with_loss, loss_fn, config=config)

        # Create a dummy output with a loss attribute
        inputs = torch.randn(10, 10).to(trainer.device)
        targets = torch.randn(10, 1).to(trainer.device)
        outputs = model_with_loss(inputs, targets)

        loss = trainer._compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar tensor
        assert torch.allclose(loss, outputs.loss)

    def test_train_step(self, model, loss_fn, config):
        """Test a single training step."""
        trainer = Trainer(model, loss_fn, config=config)

        # Create a batch
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        batch = (x, y)

        result = trainer.train_step(batch)

        assert "train_loss" in result
        assert isinstance(result["train_loss"], float)

    def test_eval_step(self, model, loss_fn, config):
        """Test a single evaluation step."""
        trainer = Trainer(model, loss_fn, config=config)

        # Create a batch
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        batch = (x, y)

        result = trainer.eval_step(batch)

        assert "eval_loss" in result
        assert isinstance(result["eval_loss"], float)

    def test_train_epoch(self, model, loss_fn, dataloader, config):
        """Test training for one epoch."""
        trainer = Trainer(model, loss_fn, config=config)

        metrics = trainer.train_epoch(dataloader)

        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], float)

    def test_train_epoch_with_eval(self, model, loss_fn, dataloader, config):
        """Test training with evaluation."""
        trainer = Trainer(model, loss_fn, config=config)

        metrics = trainer.train_epoch(dataloader, eval_dataloader=dataloader)

        assert "train_loss" in metrics
        assert "eval_loss" in metrics
        assert isinstance(metrics["train_loss"], float)
        assert isinstance(metrics["eval_loss"], float)

    def test_evaluate(self, model, loss_fn, dataloader, config):
        """Test model evaluation."""
        trainer = Trainer(model, loss_fn, config=config)

        metrics = trainer.evaluate(dataloader)

        assert "eval_loss" in metrics
        assert isinstance(metrics["eval_loss"], float)

    def test_train(self, model, loss_fn, dataloader, config, tmp_path):
        """Test full training loop."""
        # Use a temporary directory for checkpoints
        save_dir = tmp_path / "checkpoints"

        config.training.checkpoint_dir = str(save_dir)
        config.training.save_interval = 1

        trainer = Trainer(model, loss_fn, config=config)

        # Train for 2 epochs
        result = trainer.train(dataloader, num_epochs=2, eval_dataloader=dataloader)

        history = result["history"]
        assert "train_loss" in history
        assert "eval_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["eval_loss"]) == 2

    def test_save_checkpoint(self, model, loss_fn, optimizer, config, tmp_path):
        """Test saving a checkpoint."""
        # Create a trainer
        config.training.checkpoint_dir = str(tmp_path)
        trainer = Trainer(model, loss_fn, optimizer=optimizer, config=config)

        # Save a checkpoint
        trainer.epoch = 5
        trainer._save_checkpoint("checkpoint", metric_value=0.123)

        # Verify checkpoint file exists
        checkpoint_path = os.path.join(tmp_path, "checkpoint.pt")
        assert os.path.exists(checkpoint_path)

        # Load the checkpoint to verify its contents
        checkpoint = torch.load(checkpoint_path)

        # Verify the contents
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["epoch"] == 5
        assert checkpoint["metric_value"] == 0.123

    def test_load_checkpoint(self, model, loss_fn, optimizer, config, tmp_path):
        """Test loading a checkpoint."""
        # Create a checkpoint path
        config.training.checkpoint_dir = str(tmp_path)

        # Create an initial trainer and save its state
        initial_trainer = Trainer(model, loss_fn, optimizer=optimizer, config=config)

        # Change a parameter to verify it gets restored
        with torch.no_grad():
            for param in initial_trainer.model.parameters():
                param.add_(torch.ones_like(param))
                break  # Just modify one parameter

        initial_parameters = {name: param.clone() for name, param in initial_trainer.model.named_parameters()}
        initial_trainer.epoch = 10
        initial_trainer._save_checkpoint("checkpoint", metric_value=0.5)

        # Create a new trainer with the same model architecture but different parameters
        new_model = SimpleModel(input_dim=10, hidden_dim=5, output_dim=1)
        new_trainer = Trainer(new_model, loss_fn, optimizer=optim.SGD(new_model.parameters(), lr=0.01), config=config)

        # Load the checkpoint
        epoch = new_trainer.load_checkpoint("checkpoint")

        # Verify the epoch was loaded correctly
        assert new_trainer.epoch == 10

        # Verify the model parameters were restored correctly
        for name, param in new_trainer.model.named_parameters():
            assert torch.allclose(param, initial_parameters[name])

    def test_early_stopping(self, model, loss_fn, dataloader, config):
        """Test early stopping functionality."""
        # Create a trainer with early stopping
        trainer = Trainer(model, loss_fn, config=config)

        # Make the loss not improve to trigger early stopping
        # We'll do this by patching the evaluate method to always return the same loss
        original_evaluate = trainer.evaluate

        eval_counter = 0

        def mock_evaluate(dataloader):
            nonlocal eval_counter
            if eval_counter == 0:
                result = {"train_loss": 0.5}
            else:
                result = {"train_loss": 0.6}  # Worse loss, should trigger early stopping
            eval_counter += 1
            return result

        trainer.evaluate = mock_evaluate

        # Train with early stopping patience of 1
        result = trainer.train(
            dataloader,
            num_epochs=10,  # We expect to stop before this
            eval_dataloader=dataloader,
            early_stopping_patience=1,
        )

        # Should have stopped after 2 epochs (1 for initial evaluation, 1 for patience)
        assert len(result["history"]["train_loss"]) <= 3

        # Restore original method
        trainer.evaluate = original_evaluate
