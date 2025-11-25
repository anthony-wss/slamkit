"""
Evaluate a trained model on ARC-Easy dataset.

This script evaluates text-only SFT models on the ARC-Easy multiple choice task
using categorical evaluation (efficient batch processing with logit extraction).

Usage:
    # Evaluate on test set
    python example_arc/eval_arc.py \
        --model_path path/to/model \
        --data_path example_arc/arc_easy_test.jsonl \
        --batch_size 8

    # Quick evaluation on subset
    python example_arc/eval_arc.py \
        --model_path path/to/model \
        --data_path example_arc/arc_easy_test.jsonl \
        --max_problems 100
"""

import json
import argparse
import torch
import sys
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm

# Add slamkit to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from slamkit.model import UnitLM


def evaluate_arc_easy(
    model_path: str,
    data_path: str,
    batch_size: int = 8,
    max_problems: int = None,
    device: str = "cuda"
):
    """
    Evaluate a model on ARC-Easy using categorical evaluation.

    Args:
        model_path: Path to trained model checkpoint
        data_path: Path to test data jsonl file
        batch_size: Number of problems per batch
        max_problems: Limit evaluation to first N problems (None = all)
        device: Device to run evaluation on
    """
    print("=" * 60)
    print("ARC-Easy Evaluation")
    print("=" * 60)

    # Load model and tokenizer
    print(f"\nLoading model from {model_path}")

    # Load UnitLM model (custom SlamKit model)
    model = UnitLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
    ).to(device)
    model.eval()

    # Load tokenizer from the base model config
    # The UnitLM config contains the base_model_name which has the tokenizer
    base_model_name = model.config.base_model_name
    print(f"✓ Loading tokenizer from base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # Add special tokens that were used during training
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    print(f"✓ Model loaded on {device}")

    # Load test data
    print(f"\nLoading test data from {data_path}")
    with open(data_path, 'r') as f:
        test_data = [json.loads(line) for line in f]

    num_problems = len(test_data) if max_problems is None else min(len(test_data), max_problems)
    print(f"✓ Loaded {num_problems} test problems")

    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_tokens[0] if newline_tokens else tokenizer.convert_tokens_to_ids("<0x0A>")

    # Prepare for evaluation
    num_correct = 0
    total = 0

    # Cache letter token IDs
    letter_to_id = {}
    for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) == 1:
            letter_to_id[letter] = encoded[0]

    print(f"\n✓ Letter token IDs: {letter_to_id}")

    # Process in batches
    print(f"\nEvaluating {num_problems} problems in batches of {batch_size}...")
    num_batches = (num_problems + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, num_problems)
        batch_data = test_data[batch_start:batch_end]

        # Prepare batch
        prompts = []
        answer_keys = []
        available_letters_list = []

        for sample in batch_data:
            user_text = sample['user_text']
            assistant_text = sample['assistant_text']  # Ground truth answer

            # Extract available letters from the question
            # Format: "- choice text=LETTER\n"
            lines = user_text.split('\n')
            available_letters = []
            for line in lines:
                if line.startswith('-') and '=' in line:
                    letter = line.split('=')[-1].strip()
                    available_letters.append(letter)

            # Build ChatML prompt (everything up to assistant response)
            # <|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n
            prompt_ids = (
                [im_start_id] +
                tokenizer("user", add_special_tokens=False)['input_ids'] +
                [newline_id] +
                tokenizer(user_text, add_special_tokens=False)['input_ids'] +
                [im_end_id, newline_id] +
                [im_start_id] +
                tokenizer("assistant", add_special_tokens=False)['input_ids'] +
                [newline_id]
            )

            prompts.append(prompt_ids)
            answer_keys.append(assistant_text)
            available_letters_list.append(available_letters)

        # Pad prompts to same length
        max_length = max(len(p) for p in prompts)
        answer_positions = [len(p) - 1 for p in prompts]  # Position where model predicts answer
        padded_prompts = [p + [tokenizer.pad_token_id] * (max_length - len(p)) for p in prompts]
        prompt_tensor = torch.tensor(padded_prompts, dtype=torch.long, device=device)

        # Get logits
        with torch.no_grad():
            outputs = model(prompt_tensor)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        # Evaluate each problem in the batch
        for idx in range(len(batch_data)):
            answer_pos = answer_positions[idx]
            available_letters = available_letters_list[idx]
            ground_truth = answer_keys[idx]

            # Get letter token IDs for this problem
            letter_ids = [letter_to_id[letter] for letter in available_letters if letter in letter_to_id]

            if len(letter_ids) == 0:
                print(f"\nWarning: No valid letter tokens found for problem {batch_start + idx}")
                continue

            # Extract logits at answer position for valid letters
            answer_logits = logits[idx, answer_pos, letter_ids]

            # Get predicted letter
            pred_idx = answer_logits.argmax().item()
            predicted_letter = available_letters[pred_idx]

            # Check correctness
            is_correct = (predicted_letter == ground_truth)
            num_correct += int(is_correct)
            total += 1

    # Calculate accuracy
    accuracy = num_correct / total if total > 0 else 0.0

    print("\n" + "=" * 60)
    print(f"Results:")
    print(f"  Correct: {num_correct}/{total}")
    print(f"  Accuracy: {accuracy:.4f} ({100 * accuracy:.2f}%)")
    print(f"  Random baseline: 0.25 (25%)")
    print("=" * 60)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on ARC-Easy")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to test data jsonl")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--max_problems", type=int, default=None, help="Max problems to evaluate (None = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    evaluate_arc_easy(
        model_path=args.model_path,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_problems=args.max_problems,
        device=args.device
    )
