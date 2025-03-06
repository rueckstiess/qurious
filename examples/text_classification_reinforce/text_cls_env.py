import random
from typing import Dict, List, Optional

from qurious.rl.environments.environment import Environment


class TextClassificationEnvironment(Environment):
    """
    A simple environment for text classification tasks.

    The environment presents text examples that need to be classified,
    and gives rewards for correct classifications.
    """

    def __init__(
        self,
        examples: List[Dict[str, str]],
        validation_examples: Optional[List[Dict[str, str]]] = None,
        classes: Optional[List[str]] = None,
    ):
        """
        Initialize a text classification environment.

        Args:
            examples: List of {"text": "...", "label": "..."} dictionaries
            validation_examples: Optional validation set (same format as examples)
            classes: List of valid class labels (extracted from examples if None)
        """
        super().__init__()
        self.examples = examples
        self.validation_examples = validation_examples or []

        # Extract classes if not provided
        if classes is None:
            self.classes = sorted(list(set(ex["label"] for ex in examples)))
        else:
            self.classes = classes

        # Current state
        self._current_example = None
        self._done = False
        self._current_index = 0
        self._validation_mode = False

        # Shuffle examples
        self._shuffle_examples()

    def _shuffle_examples(self):
        """Shuffle the training examples."""
        if not self._validation_mode:
            random.shuffle(self.examples)

    def reset(self):
        """
        Reset the environment to a new example.

        Returns:
            The initial state (text to classify)
        """
        self._done = False

        if self._validation_mode:
            # In validation mode, go through examples sequentially
            if self._current_index >= len(self.validation_examples):
                self._current_index = 0
            self._current_example = self.validation_examples[self._current_index]
        else:
            # In training mode, use shuffled examples
            if self._current_index >= len(self.examples):
                self._current_index = 0
                self._shuffle_examples()
            self._current_example = self.examples[self._current_index]

        self._current_index += 1
        self._info = {"label": self._current_example["label"]}

        return self.get_state()

    def get_state(self):
        """
        Get the current state (text to classify).

        Returns:
            The text to classify
        """
        if self._current_example is None:
            return None
        return self._current_example["text"]

    def step(self, action: str):
        """
        Take an action (make a classification) and observe the result.

        Args:
            action: The predicted class label

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self._done:
            return self.get_state(), 0, True, self._info

        # Check if action matches the correct label
        correct_label = self._current_example["label"]
        is_correct = action.strip().lower() == correct_label.lower()

        # Reward is 1 for correct classification, 0 otherwise
        reward = 1.0 if is_correct else 0.0

        # Classification task is one-step, so always done after action
        self._done = True

        # Add info about the result
        self._info.update({"correct": is_correct, "predicted": action, "true_label": correct_label})

        return self.get_state(), reward, self._done, self._info

    def close(self):
        """Clean up resources."""
        pass

    def set_validation_mode(self, validation_mode: bool = True):
        """
        Set whether to use the validation set.

        Args:
            validation_mode: Whether to use validation examples
        """
        self._validation_mode = validation_mode
        self._current_index = 0

    def get_accuracy(self):
        """
        Calculate accuracy on the validation set.

        Returns:
            float: Accuracy (0-1)
        """
        if not self.validation_examples:
            return None

        was_in_validation = self._validation_mode
        self.set_validation_mode(True)

        correct = 0
        total = len(self.validation_examples)

        for _ in range(total):
            state = self.reset()
            action = self._get_baseline_prediction(state)
            _, _, _, info = self.step(action)
            if info["correct"]:
                correct += 1

        # Restore previous mode
        self.set_validation_mode(was_in_validation)

        return correct / total

    def _get_baseline_prediction(self, text):
        """
        Simple baseline prediction (most common class).

        Args:
            text: The text to classify

        Returns:
            str: The predicted class
        """
        # Count labels in training data
        label_counts = {}
        for ex in self.examples:
            label = ex["label"]
            label_counts[label] = label_counts.get(label, 0) + 1

        # Return most common label
        return max(label_counts.items(), key=lambda x: x[1])[0]

    @property
    def action_space(self):
        """
        Get the action space (list of valid class labels).

        Returns:
            List of valid class labels
        """
        return self.classes

    @property
    def observation_space(self):
        """
        Get the observation space.

        Returns:
            str: Description of observation space
        """
        return "text"


# Example usage:
def create_sentiment_dataset(n_examples=100):
    """
    Create a simple sentiment analysis dataset.

    Args:
        n_examples: Number of examples to generate

    Returns:
        tuple: (train_examples, validation_examples)
    """
    # Simple templates for positive and negative reviews
    positive_templates = [
        "I really enjoyed this {item}. It was {positive_adj}.",
        "This {item} is {positive_adj} and worth every penny.",
        "The {item} exceeded my expectations. {positive_adj} experience!",
        "Highly recommend this {item}. It's truly {positive_adj}.",
        "This {item} is the best I've used. {positive_adj}!",
    ]

    negative_templates = [
        "I was disappointed with this {item}. It was {negative_adj}.",
        "This {item} is {negative_adj} and not worth the money.",
        "The {item} fell short of my expectations. {negative_adj} experience!",
        "I cannot recommend this {item}. It's just {negative_adj}.",
        "This {item} is the worst I've used. {negative_adj}!",
    ]

    items = ["product", "book", "movie", "restaurant", "hotel", "service", "app", "device", "experience", "course"]

    positive_adjs = [
        "amazing",
        "excellent",
        "fantastic",
        "wonderful",
        "outstanding",
        "incredible",
        "superb",
        "brilliant",
        "terrific",
        "exceptional",
    ]

    negative_adjs = [
        "terrible",
        "awful",
        "disappointing",
        "poor",
        "mediocre",
        "horrible",
        "subpar",
        "frustrating",
        "underwhelming",
        "dreadful",
    ]

    examples = []
    for _ in range(n_examples):
        if random.random() < 0.5:
            # Generate positive example
            template = random.choice(positive_templates)
            text = template.format(item=random.choice(items), positive_adj=random.choice(positive_adjs))
            label = "positive"
        else:
            # Generate negative example
            template = random.choice(negative_templates)
            text = template.format(item=random.choice(items), negative_adj=random.choice(negative_adjs))
            label = "negative"

        examples.append({"text": text, "label": label})

    # Split into train and validation
    random.shuffle(examples)
    split_idx = int(0.8 * n_examples)
    train_examples = examples[:split_idx]
    validation_examples = examples[split_idx:]

    return train_examples, validation_examples


if __name__ == "__main__":
    train_examples, validation_examples = create_sentiment_dataset(100)
    env = TextClassificationEnvironment(train_examples, validation_examples)
    print("Train examples:", train_examples)
    print("Validation examples:", validation_examples)
    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)
    print("First example:", env.reset())
    print("First example text:", env.get_state())
