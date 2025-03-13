import torch
from tqdm.auto import tqdm

from .grid_world import make_grid_world


def extract_actions_from_responses(response):
    """
    Extract actions (e.g. "up down left right") from model responses.

    Args:
        response (str): model response (predicted actions)

    Returns:
        List of predicted actions
    """
    allowed_actions = {"up": 0, "right": 1, "down": 2, "left": 3}

    response = response.strip().lower()
    actions = response.split(" ")
    actions = [action.strip() for action in actions]

    # filter out invalid actions
    actions = [a for a in actions if a in allowed_actions.keys()]
    numeric_actions = [allowed_actions[a] for a in actions]

    return actions, numeric_actions


def run_actions_in_env(example, numeric_actions):
    """
    Generates a maze from the example and runs the actions in the environment.
    Args:
        example: maze example
        numeric_actions: List of predicted actions
    Returns:
        True if goal was reached, False otherwise
    """
    # make environment
    env = make_grid_world(
        example["size"],
        start_pos=example["start_pos"],
        goal_pos=[tuple(example["goal_pos"])],
        obstacles=example["obstacles"],
    )

    for a in numeric_actions:
        _, _, done, _ = env.step(a)
        if done:
            break

    # check if goal was reached
    return env.position == tuple(env.goal_pos[0])


def calculate_accuracy(test_data, predictions):
    """
    Calculate accuracy of predicted actions.

    Args:
        test_data: maze test data
        predictions: List of predicted actions

    Returns:
        Accuracy score
    """

    correct = 0

    for example, prediction in zip(test_data, predictions):
        # Extract actions from model response
        _, numeric_actions = extract_actions_from_responses(prediction)

        # Run actions in environment
        solved = run_actions_in_env(example, numeric_actions)
        if solved:
            correct += 1

    accuracy = correct / len(test_data)
    return accuracy


def print_predictions(preds, examples, num_examples=10):
    # Print some examples
    for i in range(min(num_examples, len(preds))):
        print(f"Example {i + 1}:")
        print(f"  Maze:\n{examples[i]['env']}")
        print(f"  Predicted: {preds[i]}")
        print(f"  Actual: {examples[i]['actions']}")
        print("")


# Evaluate on test data using batching for speed
def evaluate_model(model, tokenizer, test_dataset, max_samples=None, batch_size=8):
    model.eval()

    predictions = []

    # Sample test data
    if max_samples is not None:
        test_dataset = test_dataset.shuffle().select(range(max_samples))

    # Process test data in batches, use tqdm for progress bar
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating Accuracy"):
        batch = test_dataset.select(range(i, min(len(test_dataset), i + batch_size)))
        batch_prompts = []

        # Prepare batch inputs
        if "messages" in batch.column_names:
            # For chat-based models, use the chat template
            for item in batch:
                prompt = tokenizer.apply_chat_template(
                    item["messages"][:-1],  # Exclude assistant message
                    tokenize=False,
                    add_generation_prompt=True,
                )
                batch_prompts.append(prompt)
        else:
            # For non-chat models, use the prompts directly
            batch_prompts = batch["prompt"]

        # Tokenize all prompts in batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)

        # Generate responses for entire batch
        with torch.no_grad():
            batch_outputs = model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=20,
                do_sample=False,  # Deterministic generation
                temperature=None,  # Explicitly unset temperature for deterministic generation
                top_p=None,  # Explicitly unset top_p for deterministic generation
                top_k=None,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Process each item in the batch
        for j, (input_ids, output_ids) in enumerate(zip(batch_inputs["input_ids"], batch_outputs)):
            # Get length of input to extract only the generated part
            input_length = len(input_ids)
            generated = output_ids[input_length:]

            # Decode and extract actions
            prediction = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(prediction)

    # Calculate and return accuracy
    accuracy = calculate_accuracy(test_dataset, predictions)

    return accuracy, predictions
