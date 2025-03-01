import torch
from tqdm import tqdm
import json


def extract_actions_from_responses(tokenizer, responses):
    """
    Extract actions (up, down, left, right) from model responses.

    Args:
        tokenizer: HF tokenizer
        responses: List of tokenized responses

    Returns:
        List of predicted actions
    """
    decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
    actions = []

    for response in decoded_responses:
        response = response.strip().lower()
        if "up" in response:
            actions.append("up")
        elif "down" in response:
            actions.append("down")
        elif "left" in response:
            actions.append("left")
        elif "right" in response:
            actions.append("right")
        else:
            actions.append("unknown")

    return actions


def calculate_accuracy(predictions, references):
    """
    Calculate accuracy of predicted actions.

    Args:
        predictions: List of predicted actions
        references: List of reference actions

    Returns:
        Accuracy score
    """
    correct = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    return correct / len(references)


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
                    "content": "You are a navigation assistant. Given a maze representation, output only the next action: up, down, left, or right. "
                    "If the goal is unreachable, output 'goal unreachable'.",
                },
                {"role": "user", "content": example["env"]},
                {"role": "assistant", "content": example["action"]},
            ]
        }
        for example in maze_data
    ]

    return conversations


# Evaluate on test data using batching for speed
def evaluate_model(model, tokenizer, test_data, batch_size=8):
    model.eval()

    predictions = []
    references = []

    # Process test data in batches, use tqdm for progress bar
    for i in tqdm(range(0, len(test_data), batch_size), desc="Evaluating"):
        batch = test_data[i : i + batch_size]
        batch_prompts = []
        batch_refs = []

        # Prepare batch inputs
        for item in batch:
            prompt = tokenizer.apply_chat_template(
                item["messages"][:-1],  # Exclude assistant message
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_prompts.append(prompt)
            batch_refs.append(item["messages"][-1]["content"].strip().lower())

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

            # Decode and extract action
            prediction = extract_actions_from_responses(tokenizer, [generated])[0]

            predictions.append(prediction)
            references.append(batch_refs[j])

    # Calculate and return accuracy
    accuracy = calculate_accuracy(predictions, references)
    return accuracy, predictions, references
