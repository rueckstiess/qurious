import os

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qurious.config import TrainingConfig
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
def train_config():
    """Fixture for training configuration."""
    # Create a standard TrainingConfig and extend it with additional attributes
    config = TrainingConfig(learning_rate=0.01)
    # Add custom attributes needed for tests
    config.max_grad_norm = 1.0
    config.scheduler_step_per_batch = True
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
    def test_initialization(self, model, loss_fn, train_config):
        """Test trainer initialization."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        assert trainer.model == model
        assert trainer.loss_fn == loss_fn
        assert trainer.train_config == train_config
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.scheduler is None

        # Test device assignment
        assert isinstance(trainer.device, torch.device)

    def test_custom_optimizer(self, model, loss_fn, optimizer, train_config):
        """Test trainer with custom optimizer."""
        trainer = Trainer(model, loss_fn, optimizer=optimizer, train_config=train_config)

        assert trainer.optimizer == optimizer

    def test_get_default_optimizer(self, model, loss_fn, train_config):
        """Test the default optimizer creation."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        default_optimizer = trainer._get_default_optimizer(0.001)
        assert isinstance(default_optimizer, torch.optim.AdamW)
        assert default_optimizer.param_groups[0]["lr"] == 0.001

    def test_prepare_batch_tensor(self, model, loss_fn, train_config):
        """Test preparing a tensor batch."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Test with a single tensor
        batch = torch.randn(10, 10)
        inputs, targets = trainer._prepare_batch(batch)

        assert str(inputs.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert targets is None

    def test_prepare_batch_tuple(self, model, loss_fn, train_config):
        """Test preparing a tuple batch."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Test with a tuple of tensors
        inputs_raw = torch.randn(10, 10)
        targets_raw = torch.randn(10, 1)
        batch = (inputs_raw, targets_raw)

        inputs, targets = trainer._prepare_batch(batch)

        assert str(inputs.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert str(targets.device).split(":")[0] == str(trainer.device).split(":")[0]
        assert torch.allclose(inputs.cpu(), inputs_raw)
        assert torch.allclose(targets.cpu(), targets_raw)

    def test_prepare_batch_dict(self, model, loss_fn, train_config):
        """Test preparing a dict batch."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Test with a dictionary
        batch = {"input_ids": torch.randint(0, 100, (5, 10)), "attention_mask": torch.ones(5, 10)}

        inputs, targets = trainer._prepare_batch(batch)

        assert isinstance(inputs, dict)
        assert targets is None
        assert str(inputs["input_ids"].device).split(":")[0] == str(trainer.device).split(":")[0]
        assert str(inputs["attention_mask"].device).split(":")[0] == str(trainer.device).split(":")[0]

    def test_forward_pass_standard(self, model, loss_fn, train_config):
        """Test forward pass with a standard model."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        inputs = torch.randn(10, 10).to(trainer.device)
        outputs = trainer._forward_pass(inputs)

        assert outputs.shape == (10, 1)

    def test_forward_pass_dict(self, model_with_loss, loss_fn, train_config):
        """Test forward pass with dict inputs (HuggingFace-style)."""
        trainer = Trainer(model_with_loss, loss_fn, train_config=train_config)

        inputs = {"x": torch.randn(10, 10).to(trainer.device), "targets": torch.randn(10, 1).to(trainer.device)}

        outputs = trainer._forward_pass(inputs)

        assert hasattr(outputs, "logits")
        assert hasattr(outputs, "loss")
        assert outputs.loss is not None

    def test_compute_loss_standard(self, model, loss_fn, train_config):
        """Test loss computation with standard model."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        outputs = torch.randn(10, 1).to(trainer.device)
        targets = torch.randn(10, 1).to(trainer.device)

        loss = trainer._compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar tensor

    def test_compute_loss_with_model_loss(self, model_with_loss, loss_fn, train_config):
        """Test loss computation with a model that returns loss."""
        trainer = Trainer(model_with_loss, loss_fn, train_config=train_config)

        # Create a dummy output with a loss attribute
        inputs = torch.randn(10, 10).to(trainer.device)
        targets = torch.randn(10, 1).to(trainer.device)
        outputs = model_with_loss(inputs, targets)

        loss = trainer._compute_loss(outputs, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()  # Scalar tensor
        assert torch.allclose(loss, outputs.loss)

    def test_train_step(self, model, loss_fn, train_config):
        """Test a single training step."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Create a batch
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        batch = (x, y)

        result = trainer.train_step(batch)

        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_eval_step(self, model, loss_fn, train_config):
        """Test a single evaluation step."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Create a batch
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)
        batch = (x, y)

        result = trainer.eval_step(batch)

        assert "loss" in result
        assert isinstance(result["loss"], float)

    def test_train_epoch(self, model, loss_fn, dataloader, train_config):
        """Test training for one epoch."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        metrics = trainer.train_epoch(dataloader)

        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], float)

    def test_train_epoch_with_eval(self, model, loss_fn, dataloader, train_config):
        """Test training with evaluation."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        metrics = trainer.train_epoch(dataloader, eval_dataloader=dataloader)

        assert "train_loss" in metrics
        assert "eval_loss" in metrics
        assert isinstance(metrics["train_loss"], float)
        assert isinstance(metrics["eval_loss"], float)

    def test_evaluate(self, model, loss_fn, dataloader, train_config):
        """Test model evaluation."""
        trainer = Trainer(model, loss_fn, train_config=train_config)

        metrics = trainer.evaluate(dataloader)

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_train(self, model, loss_fn, dataloader, train_config, tmp_path):
        """Test full training loop."""
        # Use a temporary directory for checkpoints
        save_dir = tmp_path / "checkpoints"

        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Train for 2 epochs
        history = trainer.train(
            dataloader, num_epochs=2, eval_dataloader=dataloader, save_dir=str(save_dir), save_freq=1
        )

        assert "train_loss" in history
        assert "eval_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["eval_loss"]) == 2

        # Check that checkpoints were saved
        assert os.path.exists(save_dir / "best_model.pt")
        assert os.path.exists(save_dir / "checkpoint_epoch_1.pt")
        assert os.path.exists(save_dir / "checkpoint_epoch_2.pt")

    def test_checkpoint_save_load(self, model, loss_fn, optimizer, train_config, tmp_path):
        """Test checkpoint saving and loading."""
        save_path = tmp_path / "model_checkpoint.pt"

        # Create trainer and save checkpoint
        trainer = Trainer(model, loss_fn, optimizer=optimizer, train_config=train_config)
        trainer._save_checkpoint(str(save_path), epoch=5, metric_value=0.1)

        # Check that checkpoint exists
        assert os.path.exists(save_path)

        # Create a new trainer and load the checkpoint
        new_model = SimpleModel(input_dim=10, hidden_dim=5, output_dim=1)
        new_optimizer = optim.SGD(new_model.parameters(), lr=0.01)
        new_trainer = Trainer(new_model, loss_fn, optimizer=new_optimizer, train_config=train_config)

        checkpoint = new_trainer.load_checkpoint(str(save_path))

        # Verify checkpoint contents
        assert checkpoint["epoch"] == 5
        assert checkpoint["metric_value"] == 0.1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

    def test_early_stopping(self, model, loss_fn, dataloader, train_config):
        """Test early stopping functionality."""
        # Create a trainer with early stopping
        trainer = Trainer(model, loss_fn, train_config=train_config)

        # Make the loss not improve to trigger early stopping
        # We'll do this by patching the evaluate method to always return the same loss
        original_evaluate = trainer.evaluate

        eval_counter = 0

        def mock_evaluate(dataloader):
            nonlocal eval_counter
            if eval_counter == 0:
                result = {"loss": 0.5}
            else:
                result = {"loss": 0.6}  # Worse loss, should trigger early stopping
            eval_counter += 1
            return result

        trainer.evaluate = mock_evaluate

        # Train with early stopping patience of 1
        history = trainer.train(
            dataloader,
            num_epochs=10,  # We expect to stop before this
            eval_dataloader=dataloader,
            early_stopping_patience=1,
        )

        # Should have stopped after 2 epochs (1 for initial evaluation, 1 for patience)
        assert len(history["train_loss"]) <= 3

        # Restore original method
        trainer.evaluate = original_evaluate
