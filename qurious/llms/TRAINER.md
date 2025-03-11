# Trainer

A flexible PyTorch model trainer that is architecture-agnostic and suitable for various training scenarios.

## Overview

The `Trainer` class is designed to manage the training process for PyTorch models with minimal dependencies and maximum flexibility. It supports various architectures including Hugging Face models and custom PyTorch implementations, different loss functions, and includes features such as checkpointing, early stopping, and learning rate scheduling.

## Installation

The `Trainer` class is part of the `qurious` package. No additional installation steps are required if the package is already installed.

## Key Features

- **Architecture agnostic**: Works with both Hugging Face models and custom PyTorch models
- **Device support**: Automatic hardware detection for training on CPU, CUDA and MPS (Apple Silicon)
- **Flexible training loops**: Methods for single step, epoch, or multi-epoch training
- **Loss function flexibility**: Compatible with various loss functions (cross entropy, MSE, etc.)
- **Configuration support**: Integration with custom Config class
- **Extensible design**: Easy to subclass for custom behavior

## Basic Usage

```python
from qurious.trainer import Trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Define your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 2. Prepare your dataset and dataloader
x = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 3. Define optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# 4. Initialize trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn
)

# 5. Train the model
history = trainer.train(
    train_dataloader=dataloader,
    num_epochs=5,
    save_dir="./checkpoints"
)

print(f"Training history: {history}")
```

## API Reference

### Constructor

```python
Trainer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    config: Any = None,
    device: Optional[torch.device] = None,
    scheduler: Optional[Any] = None,
)
```

**Parameters:**

- `model`: PyTorch model to train
- `optimizer`: PyTorch optimizer
- `loss_fn`: Loss function (takes model output and targets, returns loss)
- `config`: Configuration object containing training parameters (optional)
- `device`: Device to run training on (if None, auto-detected)
- `scheduler`: Optional learning rate scheduler

### Methods

#### `train_step`

```python
train_step(batch: Any) -> Dict[str, float]
```

Perform a single training step.

**Parameters:**
- `batch`: Batch of data from the DataLoader

**Returns:**
- Dictionary with 'loss' and any other metrics

#### `eval_step`

```python
eval_step(batch: Any) -> Dict[str, float]
```

Perform a single evaluation step.

**Parameters:**
- `batch`: Batch of data from the DataLoader

**Returns:**
- Dictionary with 'loss' and any other metrics

#### `train_epoch`

```python
train_epoch(
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader] = None,
    epoch: int = 0
) -> Dict[str, float]
```

Train for one epoch.

**Parameters:**
- `train_dataloader`: DataLoader for training data
- `eval_dataloader`: Optional DataLoader for evaluation
- `epoch`: Current epoch number (for logging)

**Returns:**
- Dictionary with training (and optionally evaluation) metrics

#### `evaluate`

```python
evaluate(dataloader: DataLoader) -> Dict[str, float]
```

Evaluate the model on the given dataloader.

**Parameters:**
- `dataloader`: DataLoader for evaluation data

**Returns:**
- Dictionary with evaluation metrics

#### `train`

```python
train(
    train_dataloader: DataLoader,
    num_epochs: int,
    eval_dataloader: Optional[DataLoader] = None,
    save_dir: Optional[str] = None,
    save_freq: int = 1,
    best_model_metric: Optional[str] = 'eval_loss',
    early_stopping_patience: Optional[int] = None,
    callbacks: Optional[List[Callable]] = None
) -> Dict[str, List[float]]
```

Train the model for multiple epochs.

**Parameters:**
- `train_dataloader`: DataLoader for training data
- `num_epochs`: Number of epochs to train for
- `eval_dataloader`: Optional DataLoader for evaluation
- `save_dir`: Directory to save model checkpoints
- `save_freq`: Frequency (in epochs) to save regular checkpoints
- `best_model_metric`: Metric to use for saving the best model (default: 'eval_loss')
- `early_stopping_patience`: Number of epochs to wait for improvement before stopping
- `callbacks`: List of callback functions that take (trainer, epoch, metrics) as input

**Returns:**
- Dictionary with training history (lists of metrics per epoch)

#### `load_checkpoint`

```python
load_checkpoint(
    path: str, 
    load_optimizer: bool = True, 
    load_scheduler: bool = True
) -> Dict[str, Any]
```

Load a model checkpoint.

**Parameters:**
- `path`: Path to the checkpoint file
- `load_optimizer`: Whether to load optimizer state
- `load_scheduler`: Whether to load scheduler state

**Returns:**
- Checkpoint data dictionary

### Protected Methods for Subclassing

The following methods can be overridden in subclasses to customize behavior:

#### `_prepare_batch`

```python
_prepare_batch(batch: Any) -> Tuple
```

Prepare a batch for training by moving it to the correct device.

**Parameters:**
- `batch`: A batch from the dataloader

**Returns:**
- Tuple of (inputs, targets) after moving to device

#### `_forward_pass`

```python
_forward_pass(inputs: Any) -> Any
```

Perform a forward pass through the model.

**Parameters:**
- `inputs`: Model inputs

**Returns:**
- Model outputs

#### `_compute_loss`

```python
_compute_loss(outputs: Any, targets: Any) -> torch.Tensor
```

Compute the loss between outputs and targets.

**Parameters:**
- `outputs`: Model outputs
- `targets`: True targets

**Returns:**
- Loss tensor

#### `_save_checkpoint`

```python
_save_checkpoint(path: str, epoch: int, metric_value: Optional[float] = None) -> None
```

Save a model checkpoint.

**Parameters:**
- `path`: Path to save the checkpoint
- `epoch`: Current epoch number
- `metric_value`: Value of the metric used for saving best model

## Advanced Usage

### Custom Metrics for Classification

```python
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
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy
        }
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        steps = 0
        
        with torch.no_grad():
            for batch in dataloader:
                step_result = self.eval_step(batch)
                total_loss += step_result['loss']
                total_accuracy += step_result['accuracy']
                steps += 1
        
        return {
            'loss': total_loss / steps if steps > 0 else 0,
            'accuracy': total_accuracy / steps if steps > 0 else 0
        }
```

### Training with Callbacks

```python
def log_metrics_callback(trainer, epoch, metrics):
    """Custom callback to log metrics"""
    print(f"Epoch {epoch+1} metrics: {metrics}")
    
    # You could also log to TensorBoard or other tracking systems
    # writer.add_scalar('Loss/train', metrics['train_loss'], epoch)
    # if 'eval_loss' in metrics:
    #     writer.add_scalar('Loss/eval', metrics['eval_loss'], epoch)

# Use the callback during training
trainer.train(
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    num_epochs=10,
    callbacks=[log_metrics_callback]
)
```

### Fine-tuning Hugging Face Models

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Prepare your dataset (example)
# ... dataset preparation code ...

# Configure optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.01
)

# Initialize trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    loss_fn=None,  # Not needed as HF models calculate loss internally
)

# Train
history = trainer.train(
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    num_epochs=3,
    save_dir="./checkpoints"
)
```


## Configuration Options

The Trainer can accept a configuration object (`config`) with the following common parameters:

- `training.weight_decay`: Weight decay for regularization
- `training.max_grad_norm`: Gradient clipping value
- `training.scheduler_step_per_batch`: Whether to step the scheduler after each batch (True) or each epoch (False)

## Best Practices

1. **Device Management**: Let the Trainer handle device placement by using `_prepare_batch`
2. **Model Checkpointing**: Use the built-in checkpointing to save models regularly
3. **Early Stopping**: Use the `early_stopping_patience` parameter to prevent overfitting
4. **Custom Metrics**: Subclass the Trainer for task-specific metrics
5. **Callbacks**: Use callbacks for logging, visualization, or other side effects without modifying the trainer

## Common Issues and Solutions

### 1. Out of Memory Errors

If you encounter CUDA out of memory errors:
- Reduce batch size
- Use gradient accumulation (implement via a custom training loop)
- Use mixed precision training (implement in a subclass)

### 2. NaN Loss Values

If loss becomes NaN:
- Reduce learning rate
- Check for proper data normalization
- Add gradient clipping with `max_grad_norm` in the config

### 3. Slow Training

To improve training speed:
- Ensure you're using the correct device (CUDA/MPS)
- Optimize DataLoader with appropriate number of workers
- Consider batch size tuning (larger if possible)

## Contributing

To extend the Trainer class:
1. Subclass it for specialized behavior
2. Override methods like `_prepare_batch`, `_forward_pass`, or `_compute_loss`
3. Add new functionality through callbacks where possible

