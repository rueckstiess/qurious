import torch
from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset
from pymongo import MongoClient
from tqdm import tqdm

from qurious.rl.environments.grid_world import make_grid_world


def load_dataset(*args, **kwargs) -> Dataset:
    """
    Wrapper for Hugging Face datasets.load_dataset() that can load data from MongoDB as well.
    If type is "mongodb", it loads data from MongoDB, otherwise it passes all arguments
    to load_dataset function from Hugging Face datasets.

    Args:
        type (str): Type of data source, "mongodb" or any of the types supported by Hugging Face datasets.
        **kwargs: Additional arguments for loading data. For MongoDB, it requires "uri", "db", and "collection",
        and optionally takes "filter", "sort", "limit", "skip", and "projection".

    Returns:
        Dataset: Loaded dataset.
    """
    path = args[0]

    if path == "mongodb":
        # Extract MongoDB connection parameters
        uri = kwargs.pop("uri", "mongodb://localhost:27017/")
        db_name = kwargs.pop("db")
        collection_name = kwargs.pop("collection")

        # Extract optional parameters with defaults
        filter_dict = kwargs.pop("filter", {})
        sort = kwargs.pop("sort", None)
        limit = kwargs.pop("limit", None)
        skip = kwargs.pop("skip", None)
        projection = kwargs.pop("projection", {"_id": 0})

        # Connect to MongoDB
        client = MongoClient(uri)
        db = client[db_name]
        coll = db[collection_name]

        # Build the query
        cursor = coll.find(filter_dict, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        # Convert to list and create Dataset
        documents = list(cursor)
        client.close()

        return DatasetDict({"train": Dataset.from_list(documents)})
    else:
        # Use HuggingFace's load_dataset for other types
        return hf_load_dataset(*args, **kwargs)


def auto_device():
    """
    Automatically selects the device for PyTorch based on availability of CUDA or MPS.
    Returns:
        torch.device: The selected device (either "cuda", "mps", or "cpu").
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device


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

    print(f"\nEvaluating {len(test_dataset)} samples...")
    # Process test data in batches, use tqdm for progress bar
    for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
        batch = test_dataset.select(range(i, min(len(test_dataset), i + batch_size)))
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
    accuracy = calculate_accuracy(test_dataset, predictions)
    print_predictions(predictions, test_dataset)

    return accuracy, predictions
