import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse


def main():
    parser = argparse.ArgumentParser(description="Chat with a fine-tuned model")
    parser.add_argument("--adapter", default="./grid_world_lora_adapter", help="Path to the fine-tuned model adapter")
    parser.add_argument("--base_model", default="meta-llama/Llama-3.2-1B-Instruct", help="Base model name")
    args = parser.parse_args()

    # Load the base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load the base model first
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load the LoRA adapter on top of the base model
    if args.adapter != "none":
        print(f"Loading adapter from {args.adapter}...")
        model = PeftModel.from_pretrained(model, args.adapter)

    print("Model loaded successfully! Type 'exit' to end the conversation.")
    print("=" * 50)

    # Start chat loop
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that can both navigate mazes and answer general questions.",
        }
    ]

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        # unescape \n in user_input
        user_input = user_input.replace("\\n", "\n")

        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})
        print(messages)

        # Format conversation using the chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Check if input is likely asking for navigation help
        is_navigation = any(word in user_input.lower() for word in ["maze", "navigate", "up", "down", "left", "right"])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512 if not is_navigation else 20,  # Shorter responses for navigation
                do_sample=True if not is_navigation else False,  # Deterministic for navigation
                temperature=0.7 if not is_navigation else None,  # Use temperature for general chat
                top_p=0.9 if not is_navigation else None,  # Use top_p for general chat
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Get only the generated part of the response
        input_length = inputs.input_ids.shape[1]
        generated_output = outputs[0][input_length:]

        # Decode the response
        response = tokenizer.decode(generated_output, skip_special_tokens=True).strip()
        print(f"\nAssistant: {response}")

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
