import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .policy import Policy


def auto_device():
    """
    Automatically select the best available device.

    Returns:
        torch.device: The selected device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_chat_model(model_name: str) -> bool:
    """
    Determine if a model is likely a chat model based on its name.

    Args:
        model_name: Model name or path

    Returns:
        bool: Whether the model is likely a chat model
    """
    chat_identifiers = [
        "chat",
        "instruct",
        "dialog",
        "conversational",
        "llama-2",
        "llama2",
        "llama-3",
        "llama3",
        "mistral",
        "mixtral",
        "zephyr",
        "vicuna",
        "wizard",
    ]
    model_name_lower = model_name.lower()
    return any(identifier in model_name_lower for identifier in chat_identifiers)


def default_state_to_prompt(state: Any) -> str:
    """
    Default conversion from state to text prompt.

    Args:
        state: Environment state

    Returns:
        str: Text prompt
    """
    return f"Given the current state: {state}\nChoose the next action:"


def default_state_to_chat(state: Any) -> List[Dict[str, str]]:
    """
    Default conversion from state to chat messages.

    Args:
        state: Environment state

    Returns:
        List[Dict[str, str]]: Chat messages
    """
    return [
        {"role": "system", "content": "You are an agent that chooses actions to solve tasks."},
        {"role": "user", "content": f"Given the current state: {state}\nWhat action would you take?"},
    ]


def default_action_parser(output: str) -> str:
    """
    Default parsing of model output to action.

    Args:
        output: Model text output

    Returns:
        Action representation
    """
    # Just return the first line of the output as the action
    return output.strip().split("\n")[0].strip()


# Default generation parameters (conservative settings)
DEFAULT_GENERATION_KWARGS = {
    "max_new_tokens": 20,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": 50256,  # Default for GPT-2, will be overridden in init
}


class LLMPolicy(Policy):
    """
    Policy that uses a Language Model to generate actions based on state observations.

    This policy can work with both instruction-tuned models (using chat format) and
    standard language models (using prompt-completion format).
    """

    def __init__(
        self,
        model_name_or_path: str,
        state_formatter: Optional[Callable] = None,
        action_parser: Optional[Callable] = None,
        use_chat_format: Optional[bool] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        model_kwargs: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize an LLM-based policy.

        Args:
            model_name_or_path: Hugging Face model identifier or local path
            state_formatter: Function to convert environment state to prompt/message format
            action_parser: Function to convert LLM output to action representation
            use_chat_format: Whether to use chat format (auto-detected if None)
            tokenizer_kwargs: Additional arguments for tokenizer initialization
            model_kwargs: Additional arguments for model initialization
            generation_kwargs: Parameters for text generation (temperature, max_tokens, etc.)
            device: Device to load the model on ("cpu", "cuda", "mps", etc.)
        """
        self.model_name = model_name_or_path

        # Determine if chat format should be used
        if use_chat_format is None:
            self.use_chat_format = is_chat_model(model_name_or_path)
        else:
            self.use_chat_format = use_chat_format

        # Set state formatter based on format
        if state_formatter is None:
            self.state_formatter = default_state_to_chat if self.use_chat_format else default_state_to_prompt
        else:
            self.state_formatter = state_formatter

        # Set action parser
        self.action_parser = action_parser or default_action_parser

        # Initialize device
        self.device = torch.device(device) if device else auto_device()

        # Get default kwargs if not provided
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.model_kwargs = model_kwargs or {"torch_dtype": torch.float16}

        # Load tokenizer and model
        self._load_tokenizer_and_model()

        # Set generation parameters with reasonable defaults
        self.generation_kwargs = dict(DEFAULT_GENERATION_KWARGS)
        if generation_kwargs:
            self.generation_kwargs.update(generation_kwargs)

        # Ensure pad token is correctly set
        if self.tokenizer.pad_token_id is not None:
            self.generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id

    def _load_tokenizer_and_model(self):
        """Load the tokenizer and model from the specified path."""
        # Load tokenizer with padding token
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **self.tokenizer_kwargs)

        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = "<pad>"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=self.device, **self.model_kwargs)

    def get_action(self, state: Any) -> Any:
        """
        Generate an action for the given state using the LLM.

        Args:
            state: The current environment state

        Returns:
            The action to take (parsed according to action_parser)
        """
        # Format state based on model type
        if self.use_chat_format:
            messages = self.state_formatter(state)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.state_formatter(state)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.generation_kwargs)

        # Decode the output
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)

        # Parse the action
        action = self.action_parser(generated_text)
        return action

    def get_action_probabilities(self, state: Any) -> np.ndarray:
        """
        Get probability distribution over actions for the given state.

        Note: For LLM policies with large action spaces, this is often approximated
        or not fully implemented unless working with a small, discrete action space.

        Args:
            state: The current state

        Returns:
            numpy.ndarray: Probability distribution over actions
        """
        # This default implementation is a placeholder
        # In practice, this should be overridden by subclasses for specific action spaces
        # or could compute logprobs for particular action tokens
        raise NotImplementedError(
            "Action probabilities not directly available for LLM policy. "
            "Override this method for your specific action space."
        )

    def update(self, state: Any, action: Any, value: float):
        """
        Update the policy for a state-action pair based on a value.

        For base LLMPolicy, this is a no-op as the model isn't updated directly.
        Subclasses that support training should override this.

        Args:
            state: The state
            action: The action
            value: The value (reward) used to update the policy
        """
        # Base implementation does nothing
        # Training-enabled subclasses will override this
        pass

    def update_from_value_fn(self, value_function: Any):
        """
        Update the policy based on a value function.

        For base LLMPolicy, this is a no-op.
        Subclasses that support value-based updates should override this.

        Args:
            value_function: The value function to use for the update
        """
        # Base implementation does nothing
        pass

    def save(self, directory: str):
        """
        Save the policy model and configuration to a directory.

        Args:
            directory: Directory path to save the policy
        """
        os.makedirs(directory, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)

        # Save configuration
        config = {
            "model_name": self.model_name,
            "use_chat_format": self.use_chat_format,
            "generation_kwargs": self.generation_kwargs,
        }

        # Save config as JSON
        import json

        with open(os.path.join(directory, "llm_policy_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, directory: str, state_formatter=None, action_parser=None):
        """
        Load a policy from a directory.

        Args:
            directory: Directory path where the policy is saved
            state_formatter: Optional state formatter (uses default if None)
            action_parser: Optional action parser (uses default if None)

        Returns:
            LLMPolicy: The loaded policy
        """
        import json

        # Load configuration
        with open(os.path.join(directory, "llm_policy_config.json"), "r") as f:
            config = json.load(f)

        # Create policy instance
        policy = cls(
            model_name_or_path=directory,  # Load from local directory
            state_formatter=state_formatter,
            action_parser=action_parser,
            use_chat_format=config.get("use_chat_format"),
            generation_kwargs=config.get("generation_kwargs"),
        )

        return policy

    def train(self):
        """Set the policy to training mode."""
        self.model.train()

    def eval(self):
        """Set the policy to evaluation mode."""
        self.model.eval()


class TrainableLLMPolicy(LLMPolicy):
    """LLM Policy that can be updated through gradients."""

    def __init__(self, model_name_or_path: str, *args, **kwargs):
        """
        Initialize a trainable LLM policy.

        Args:
            model_name_or_path: Hugging Face model identifier or local path
            *args, **kwargs: Additional arguments for LLMPolicy
        """
        super().__init__(model_name_or_path, *args, **kwargs)

        # Track training metrics
        self.train_metrics = {"loss": []}

    def get_action_probabilities(self, state):
        """
        For LLM policies with large action spaces, calculating full action
        probabilities is often impractical. This method provides a warning
        and a simplified approximation.

        Args:
            state: The current state

        Returns:
            numpy.ndarray: Approximate probability distribution
        """
        warning_msg = (
            "Warning: get_action_probabilities() for LLM policies provides only "
            "a rough approximation. For LLMs with large action spaces, this "
            "should not be relied upon for algorithmic decisions."
        )
        print(warning_msg)

        # Return a placeholder distribution (just makes the highest probability
        # for the action we would actually choose)
        action = self.get_action(state)
        # Create a minimal action space with the chosen action having highest probability
        return np.array([0.1, 0.9] if action else [0.9, 0.1])

    def _get_action_logprobs(self, state, action):
        """
        Get log probabilities of an action given a state.

        This implementation treats the entire output sequence as a single action
        and calculates its log probability.

        Args:
            state: Environment state
            action: Taken action (text string)

        Returns:
            torch.Tensor: Log probability of the action
        """
        # Format state based on model type
        if self.use_chat_format:
            messages = self.state_formatter(state)
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = self.state_formatter(state)

        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Tokenize the action (output sequence)
        action_ids = self.tokenizer(action, return_tensors="pt").input_ids.to(self.device)

        # We'll calculate log probability token by token
        log_prob = 0

        # Start with the prompt
        current_input_ids = inputs.input_ids

        # For each token in the action, calculate its probability given previous context
        for i in range(action_ids.shape[1]):
            # Forward pass to get logits
            with torch.no_grad():
                outputs = self.model(input_ids=current_input_ids)

            logits = outputs.logits[:, -1, :]  # Get logits for the next token prediction

            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Get probability of the actual next token
            next_token_id = action_ids[0, i]
            next_token_prob = probs[0, next_token_id]

            # Add log probability
            log_prob += torch.log(next_token_prob + 1e-10)  # Add small epsilon for numerical stability

            # Update context with this token for next iteration
            current_input_ids = torch.cat([current_input_ids, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

        return log_prob
