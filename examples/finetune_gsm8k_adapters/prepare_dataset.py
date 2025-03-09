from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


class GSM8KDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for item in dataset:
            # Format: "Question: {question}\nAnswer: {answer}"
            text = f"Question: {item['question']}\nAnswer: {item['answer']}"

            # Tokenize
            encodings = tokenizer(
                text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt"
            )

            # Store as a training example
            self.examples.append(
                {
                    "input_ids": encodings.input_ids[0],
                    "attention_mask": encodings.attention_mask[0],
                    "labels": encodings.input_ids[0].clone(),  # For causal LM, labels = input_ids
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def prepare_dataset(tokenizer):
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and preprocess the GSM8K dataset
    train_dataset, test_dataset = load_dataset("gsm8k", "main", split=["train", "test"])

    # Create dataset and dataloader
    train_dataset = GSM8KDataset(train_dataset, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    eval_dataset = GSM8KDataset(test_dataset, tokenizer)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

    return train_dataloader, eval_dataloader
