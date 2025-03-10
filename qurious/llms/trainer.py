import datetime
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlflow
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from qurious.config import TrainingConfig
from qurious.utils import auto_device


class Trainer:
    """
    Generic PyTorch model trainer that supports various architectures and loss functions.

    Attributes:
        model: The PyTorch model to train.
        optimizer: The optimizer used for training.
        loss_fn: The loss function.
        device: The device to run training on (auto-detected if not provided).
        config: Configuration object for training parameters.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
        train_config: TrainingConfig = None,
        scheduler: Optional[Any] = None,
        loggers: Optional[List[str | Callable]] = ["console"],
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            optimizer: PyTorch optimizer (if None, uses Adam with default settings)
            loss_fn: Loss function (takes model output and targets, returns loss)
            device: Device to run training on (if None, auto-detected)
            train_config: Configuration object containing training parameters
            scheduler: Optional learning rate scheduler
        """
        self.model = model
        self.train_config = train_config
        if device is None:
            self.device = auto_device()
        elif device == "auto":
            self.device = auto_device()
        else:
            self.device = device

        # Use provided optimizer or create default one
        default_lr = 1e-4  # Default learning rate if train_config is None
        learning_rate = train_config.learning_rate if train_config is not None else default_lr
        self.optimizer = optimizer if optimizer is not None else self._get_default_optimizer(learning_rate)

        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.loggers = loggers

        if experiment_name is None:
            experiment_name = f"Experiment {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"
        self.experiment_name = experiment_name

        # set up mlflow logging if specified
        if "mlflow" in self.loggers:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run(run_name=run_name)

        # Move model to the appropriate device
        self.model.to(self.device)

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    def _get_default_optimizer(self, learning_rate: float) -> torch.optim.Optimizer:
        """
        Returns a default AdamW optimizer for the model.

        Args:
            learning_rate: Learning rate for the optimizer

        Returns:
            torch.optim.AdamW optimizer
        """
        return torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

    def _prepare_batch(self, batch: Any) -> Tuple:
        """
        Prepare a batch for training by moving it to the correct device.

        This method should be overridden for custom data formatting.

        Args:
            batch: A batch from the dataloader

        Returns:
            Tuple of (inputs, targets) after moving to device
        """
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device), None

        if isinstance(batch, (list, tuple)):
            # Assume first element is input, second is target
            if len(batch) >= 2:
                inputs = batch[0].to(self.device) if isinstance(batch[0], torch.Tensor) else batch[0]
                targets = batch[1].to(self.device) if isinstance(batch[1], torch.Tensor) else batch[1]
                return inputs, targets

        if isinstance(batch, dict):
            # Move all tensor values to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(self.device)
            return batch, None

        # If we can't determine the format, return as is
        return batch, None

    def _forward_pass(self, inputs: Any) -> Any:
        """
        Perform a forward pass through the model.

        Override this method for custom forward pass behavior.

        Args:
            inputs: Model inputs

        Returns:
            Model outputs
        """
        if isinstance(inputs, dict):
            # For HuggingFace-style models that take keyword arguments
            return self.model(**inputs)
        else:
            # For standard PyTorch models
            return self.model(inputs)

    def _compute_loss(self, outputs: Any, targets: Any) -> torch.Tensor:
        """
        Compute the loss between outputs and targets.

        Override this method for custom loss computation.

        Args:
            outputs: Model outputs
            targets: True targets

        Returns:
            Loss tensor
        """
        if hasattr(outputs, "loss") and outputs.loss is not None:
            # Handle HuggingFace-style models that return loss
            return outputs.loss

        if targets is None:
            raise ValueError("Targets are None, but model didn't return loss. Check your data pipeline.")

        # Otherwise use the provided loss function
        return self.loss_fn(outputs, targets)

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to MLFlow and/or to the console.
        If self.loggers contains a callable, it will be called with (self, metrics, step).

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if "console" in self.loggers:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.logger.info(f"Step {step}: {metrics_str}")

        if "mlflow" in self.loggers:
            mlflow.log_metrics(metrics, step=step)

        # Call any custom logger functions
        logger_fns = [logger for logger in self.loggers if isinstance(logger, Callable)]
        for logger_fn in logger_fns:
            logger_fn(self, metrics, step)

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Batch of data from the DataLoader

        Returns:
            Dict with 'loss' and any other metrics
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Prepare batch and perform forward pass
        inputs, targets = self._prepare_batch(batch)
        outputs = self._forward_pass(inputs)

        # Compute loss and backpropagate
        loss = self._compute_loss(outputs, targets)
        loss.backward()

        # Apply gradient clipping if specified in config
        if self.train_config and hasattr(self.train_config, "max_grad_norm") and self.train_config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config.max_grad_norm)

        self.optimizer.step()

        # Update learning rate if using a scheduler that steps per batch
        if (
            self.scheduler is not None
            and hasattr(self.train_config, "scheduler_step_per_batch")
            and self.train_config.scheduler_step_per_batch
        ):
            self.scheduler.step()

        return {"loss": loss.item()}

    def eval_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single evaluation step.

        Args:
            batch: Batch of data from the DataLoader

        Returns:
            Dict with 'loss' and any other metrics
        """
        self.model.eval()

        with torch.no_grad():
            # Prepare batch and perform forward pass
            inputs, targets = self._prepare_batch(batch)
            outputs = self._forward_pass(inputs)

            # Compute loss
            loss = self._compute_loss(outputs, targets)

        return {"loss": loss.item()}

    def train_epoch(
        self, train_dataloader: DataLoader, eval_dataloader: Optional[DataLoader] = None, epoch: int = 0
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation
            epoch: Current epoch number (for logging)

        Returns:
            Dict with training (and optionally evaluation) metrics
        """
        # Training phase
        self.model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            step_result = self.train_step(batch)
            train_loss += step_result["loss"]
            train_steps += 1

            # log metrics every log_interval steps
            if train_steps % self.train_config.log_interval == 0:
                self._log_metrics({"train_loss": step_result["loss"]}, train_steps)

            # Update progress bar with current loss
            pbar.set_postfix(loss=f"{step_result['loss']:.4f}")

        # Calculate average training loss
        avg_train_loss = train_loss / train_steps if train_steps > 0 else 0
        metrics = {"train_loss": avg_train_loss}

        # Evaluation phase
        if eval_dataloader is not None:
            eval_metrics = self.evaluate(eval_dataloader)
            metrics.update({f"eval_{k}": v for k, v in eval_metrics.items()})
            self._log_metrics(eval_metrics, train_steps)

        # Update learning rate scheduler if it steps per epoch
        if self.scheduler is not None and not (
            hasattr(self.train_config, "scheduler_step_per_batch") and self.train_config.scheduler_step_per_batch
        ):
            if hasattr(self.scheduler, "step_with_metrics"):
                # Some schedulers like ReduceLROnPlateau need validation metrics
                self.scheduler.step(metrics.get("eval_loss", avg_train_loss))
            else:
                self.scheduler.step()

        return metrics

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Dict with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        steps = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                step_result = self.eval_step(batch)
                total_loss += step_result["loss"]
                steps += 1

        # Calculate average evaluation loss
        avg_loss = total_loss / steps if steps > 0 else 0
        return {"loss": avg_loss}

    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 1,
        best_model_metric: Optional[str] = "eval_loss",
        early_stopping_patience: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.

        Args:
            train_dataloader: DataLoader for training data
            num_epochs: Number of epochs to train for
            eval_dataloader: Optional DataLoader for evaluation
            save_dir: Directory to save model checkpoints
            save_freq: Frequency (in epochs) to save regular checkpoints
            best_model_metric: Metric to use for saving the best model (default: 'eval_loss')
            early_stopping_patience: Number of epochs to wait for improvement before stopping
            callbacks: List of callback functions that take (trainer, epoch, metrics) as input

        Returns:
            Dict with training history (lists of metrics per epoch)
        """
        # Create save directory if needed
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Initialize tracking variables
        history = {"train_loss": []}
        if eval_dataloader is not None:
            history["eval_loss"] = []

        best_metric_value = float("inf")  # For minimization (like loss)
        best_epoch = -1
        no_improvement_count = 0

        self.logger.info(f"Starting training for {num_epochs} epochs")

        try:
            # Main training loop
            for epoch in range(num_epochs):
                # Train for one epoch and collect metrics
                epoch_metrics = self.train_epoch(train_dataloader, eval_dataloader, epoch)

                # Update history
                for key, value in epoch_metrics.items():
                    if key not in history:
                        history[key] = []
                    history[key].append(value)

                # Check for best model
                current_metric = epoch_metrics.get(best_model_metric, epoch_metrics.get("train_loss"))
                is_improvement = False

                if best_model_metric.endswith("loss"):  # Minimize loss
                    if current_metric < best_metric_value:
                        best_metric_value = current_metric
                        best_epoch = epoch
                        is_improvement = True

                        # Save best model
                        if save_dir:
                            self._save_checkpoint(os.path.join(save_dir, "best_model.pt"), epoch, best_metric_value)
                            self.logger.info(f"Best model saved at epoch {epoch + 1}")
                else:  # Maximize other metrics (accuracy, f1, etc.)
                    if current_metric > best_metric_value:
                        best_metric_value = current_metric
                        best_epoch = epoch
                        is_improvement = True

                        # Save best model
                        if save_dir:
                            self._save_checkpoint(os.path.join(save_dir, "best_model.pt"), epoch, best_metric_value)
                            self.logger.info(f"Best model saved at epoch {epoch + 1}")

                # Regular checkpoint saving
                if save_dir and save_freq > 0 and (epoch + 1) % save_freq == 0:
                    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pt")
                    self._save_checkpoint(checkpoint_path, epoch)
                    self.logger.info(f"Checkpoint saved at epoch {epoch + 1}")

                # Early stopping check
                if early_stopping_patience is not None:
                    if is_improvement:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= early_stopping_patience:
                        self.logger.info(
                            f"Early stopping triggered after {epoch + 1} epochs. Best epoch was {best_epoch + 1}."
                        )
                        break

                # Execute callbacks
                if callbacks:
                    for callback in callbacks:
                        callback(self, epoch, epoch_metrics)

            self.logger.info(f"Training completed. Best {best_model_metric} was at epoch {best_epoch + 1}")

        except KeyboardInterrupt:
            self.logger.info("Training interrupted.")
        except Exception as e:
            self.logger.error(f"An error occurred during training: {e}")
            if "mlflow" in self.loggers:
                mlflow.log_param("error", str(e))
            raise
        finally:
            # Ensure MLFlow run is ended
            if "mlflow" in self.loggers:
                mlflow.end_run()
            return history

    def _save_checkpoint(self, path: str, epoch: int, metric_value: Optional[float] = None) -> None:
        """
        Save a model checkpoint.

        Args:
            path: Path to save the checkpoint
            epoch: Current epoch number
            metric_value: Value of the metric used for saving best model
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.train_config,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if metric_value is not None:
            checkpoint["metric_value"] = metric_value

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, load_optimizer: bool = True, load_scheduler: bool = True) -> Dict[str, Any]:
        """
        Load a model checkpoint.

        Args:
            path: Path to the checkpoint file
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state

        Returns:
            Checkpoint data dictionary
        """
        # Note: In a production environment, you might want to use weights_only=True for security
        # but this requires additional configuration with torch.serialization.add_safe_globals
        with torch.serialization.safe_globals([TrainingConfig]):
            checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_scheduler and self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return checkpoint
