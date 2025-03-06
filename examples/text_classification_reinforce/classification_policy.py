from typing import List, Optional

from qurious.rl.policies.llm_policy import TrainableLLMPolicy, is_chat_model


def classification_prompt_formatter(text, classes=None):
    """
    Format text classification task as a prompt.

    Args:
        text: The text to classify
        classes: Optional list of class labels

    Returns:
        str: Formatted prompt
    """
    base_prompt = "Classify the following text as "

    if classes:
        class_options = " or ".join([f'"{c}"' for c in classes])
        base_prompt += f"{class_options}.\n\n"
    else:
        base_prompt += "positive or negative.\n\n"

    base_prompt += f'Text: "{text}"\n\nClassification:'

    return base_prompt


def classification_chat_formatter(text, classes=None):
    """
    Format text classification task as chat messages.

    Args:
        text: The text to classify
        classes: Optional list of class labels

    Returns:
        List of message dictionaries
    """
    if classes:
        class_options = " or ".join([f'"{c}"' for c in classes])
        system_content = f"You are a text classifier. Classify text as {class_options}. Respond with ONLY the classification label, nothing else."
    else:
        system_content = 'You are a sentiment analyzer. Classify text as either "positive" or "negative". Respond with ONLY the classification label, nothing else.'

    return [{"role": "system", "content": system_content}, {"role": "user", "content": text}]


def classification_action_parser(output):
    """
    Parse model output to extract class label.

    Args:
        output: Model text output

    Returns:
        str: Extracted class label
    """
    # Strip whitespace and return first line only
    return output.strip().split("\n")[0].strip().lower()


class ClassificationPolicy(TrainableLLMPolicy):
    """
    LLM policy specialized for text classification tasks.
    """

    def __init__(self, model_name_or_path: str, classes: Optional[List[str]] = None, **kwargs):
        """
        Initialize a classification policy.

        Args:
            model_name_or_path: Hugging Face model identifier or local path
            classes: List of valid class labels
            **kwargs: Additional arguments for TrainableLLMPolicy
        """
        self.classes = classes or ["positive", "negative"]

        # Create state formatter with class information
        if kwargs.get("use_chat_format", None) or is_chat_model(model_name_or_path):
            state_formatter = lambda text: classification_chat_formatter(text, self.classes)
        else:
            state_formatter = lambda text: classification_prompt_formatter(text, self.classes)

        # Initialize with classification-specific formatters and parsers
        super().__init__(
            model_name_or_path=model_name_or_path,
            state_formatter=state_formatter,
            action_parser=classification_action_parser,
            **kwargs,
        )

        # Set shorter generation parameters for classification
        self.generation_kwargs.update(
            {
                "max_new_tokens": 5,  # Classification labels are short
                "temperature": 0.3,  # Lower temperature for more deterministic outputs
            }
        )
