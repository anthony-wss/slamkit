"""
Prepare ARC-Easy dataset for text-only SFT training.

This script downloads ARC-Easy from HuggingFace and converts it to
the SFT format used by SlamKit (text-only, no audio).

Usage:
    python example_arc/prepare_arc_data.py

Output:
    - example_arc/arc_easy_train.jsonl: Training data (~2.3K examples)
    - example_arc/arc_easy_val.jsonl: Validation data
    - example_arc/arc_easy_test.jsonl: Test data (~2.4K examples)
"""

import json
import os
from pathlib import Path
from datasets import load_dataset


def render_multiple_choice(question: str, choices: list, labels: list) -> str:
    """
    Render a multiple choice question in the format expected by the model.

    Based on nanochat's design:
    - Letter comes AFTER the choice text for better token binding
    - No whitespace before the letter (critical for tokenization)
    """
    prompt = f"Multiple Choice question: {question}\n"
    for choice, label in zip(choices, labels):
        prompt += f"- {choice}={label}\n"
    prompt += "\nRespond only with the letter of the correct answer."
    return prompt


def prepare_arc_easy(output_dir: str = "example_arc"):
    """
    Download and prepare ARC-Easy dataset for SFT training.

    Converts HuggingFace dataset to text-only SFT format with user/assistant messages.
    """
    print("=" * 60)
    print("Preparing ARC-Easy Dataset for SFT Training")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ARC-Easy dataset
    print("\nDownloading ARC-Easy dataset from HuggingFace...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy")

    print(f"✓ Dataset loaded")
    print(f"  - Train: {len(dataset['train'])} examples")
    print(f"  - Validation: {len(dataset['validation'])} examples")
    print(f"  - Test: {len(dataset['test'])} examples")

    # Process each split
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]

        output_file = os.path.join(output_dir, f"arc_easy_{split_name}.jsonl")

        print(f"\nProcessing {split_name} split...")
        with open(output_file, 'w') as f:
            for idx, example in enumerate(split_data):
                # Extract fields
                question = example['question']
                choices = example['choices']['text']
                labels = example['choices']['label']
                answer_key = example['answerKey']

                # Verify answer key is valid
                assert answer_key in labels, f"Answer key {answer_key} not in labels {labels}"

                # Format as multiple choice prompt
                user_message = render_multiple_choice(question, choices, labels)
                assistant_message = answer_key  # Just the letter: "A", "B", "C", or "D"

                # Create conversation in text-only SFT format
                conversation = {
                    "user_text": user_message,
                    "assistant_text": assistant_message,
                }

                # Write to file
                f.write(json.dumps(conversation) + '\n')

        print(f"  ✓ Saved {len(split_data)} examples to {output_file}")

    print("\n" + "=" * 60)
    print("✓ Dataset preparation complete!")
    print("=" * 60)

    # Show example
    print("\nExample conversation:")
    print("-" * 60)
    with open(os.path.join(output_dir, "arc_easy_train.jsonl"), 'r') as f:
        example = json.loads(f.readline())
        print("USER:")
        print(example['user_text'])
        print("\nASSISTANT:")
        print(example['assistant_text'])
    print("-" * 60)


if __name__ == "__main__":
    prepare_arc_easy()
