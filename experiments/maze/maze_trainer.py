import torch

from qurious.llms.trainer import Trainer


class MazeTrainer(Trainer):
    def eval_step(self, batch):
        inputs, targets = self._prepare_batch(batch)
        outputs = self._forward_pass(inputs)
        loss = self._compute_loss(outputs, targets)

        # Turn outputs into tokens
        # ...

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        return {"eval_loss": loss.item(), "accuracy": accuracy}

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                step_metrics = self.eval_step(batch)
                total_loss += step_metrics["eval_loss"]
                total_accuracy += step_metrics["accuracy"]
                steps += 1

        return {
            "eval_loss": total_loss / steps if steps > 0 else 0,
            "accuracy": total_accuracy / steps if steps > 0 else 0,
        }
