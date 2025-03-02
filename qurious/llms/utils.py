import torch
from tqdm import tqdm
import json
import random
from qurious.environments import GridWorld


def make_env(size, **kwargs):
    # create random start position in grid
    random_start_pos = (random.randint(0, size - 1), random.randint(0, size - 1))

    # create random goal position in grid (different from start)
    while True:
        random_goal_pos = (random.randint(0, size - 1), random.randint(0, size - 1))
        if random_goal_pos != random_start_pos:
            break

    # Create a maze with a guaranteed path
    env = GridWorld(
        width=size,
        height=size,
        start_pos=kwargs.get("start_pos", random_start_pos),
        goal_pos=[kwargs.get("goal_pos", random_goal_pos)],
        obstacles=kwargs.get("obstacles", 0.2),
        terminal_reward=0.0,
        step_penalty=0.1,
        max_steps=100,
    )
    return env


def extract_actions_from_responses(response):
    """
    Extract actions (up, down, left, right) from model responses.

    Args:
        response (str): model response (predicted actions)

    Returns:
        List of predicted actions
    """
    allowed_actions = {"up": 0, "right": 1, "down": 2, "left": 3}

    response = response.strip().lower()
    actions = response.split(",")
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
    env = make_env(
        example["size"], start_pos=example["start_pos"], goal_pos=example["goal_pos"], obstacles=example["obstacles"]
    )

    for a in numeric_actions:
        _, _, done, _ = env.step(a)
        if done:
            break

    # check if goal was reached
    return env.position == env.goal_pos[0]


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


def load_maze_data(filename):
    """
    Load maze conversation data.

    Returns:
        List of dicts with 'system', 'user', 'assistant
    """

    # load json into list of dicts
    with open(filename, "r") as f:
        maze_data = json.load(f)

    conversations = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a navigation assistant. Given a maze representation, output the actions "
                    "to move the agent (A) to the goal (G), avoiding obstacles (#). Output actions as a "
                    "comma-separated list of up, down, left, right. Do not include any other text.",
                },
                {"role": "user", "content": example["env"]},
                {"role": "assistant", "content": example["actions"]},
            ],
            **example,
        }
        for example in maze_data
    ]

    return conversations


# Evaluate on test data using batching for speed
def evaluate_model(model, tokenizer, test_data, batch_size=8):
    model.eval()

    predictions = []

    # Process test data in batches, use tqdm for progress bar
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i : i + batch_size]
        batch_prompts = []

        # Prepare batch inputs
        for item in batch:
            prompt = tokenizer.apply_chat_template(
                item["messages"][:-1],  # Exclude assistant message
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_prompts.append(prompt)

        # Tokenize all prompts in batch
        batch_inputs = tokenizer(batch_prompts, padding=True, return_tensors="pt").to(model.device)

        # Generate responses for entire batch
        with torch.no_grad():
            batch_outputs = model.generate(
                input_ids=batch_inputs["input_ids"],
                attention_mask=batch_inputs["attention_mask"],
                max_new_tokens=200,
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
    accuracy = calculate_accuracy(test_data, predictions)
    return accuracy, predictions
