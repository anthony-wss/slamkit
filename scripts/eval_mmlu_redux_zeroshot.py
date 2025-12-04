#!/usr/bin/env python3
"""
Zero-shot MMLU accuracy evaluation for SlamKit model checkpoints.

Evaluates standard MMLU performance (predicting correct answer from A/B/C/D).
Supports both standard HuggingFace models and SlamKit UnitLM models.

Usage:
    python scripts/eval_mmlu_redux_zeroshot.py \
        --model_path example_slamomni/pretrain_qwen3-0.6B_checkpoint \
        --config abstract_algebra \
        --output_dir outputs/mmlu_redux_eval \
        --bf16
"""

import argparse
import sys
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add parent dir to path for slamkit imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from slamkit.model import UnitLM

# Borrow from nanochat
def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.

    Note two important design decisions:
    1)
    Bigger models don't care as much, but smaller models prefer to have
    the letter *after* the choice, which results in better binding.
    2)
    There is no whitespace between the delimiter (=) and the letter.
    This is actually critical because the tokenizer has different token ids
    for " A" vs. "A". The assistant responses will be just the letter itself,
    i.e. "A", so it is important that here in the prompt it is the exact same
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    about this too much, but smaller models do care about some of these details.
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


def load_slamkit_model(model_path, tokenizer_name, device, use_bf16=False):
    """
    Load a SlamKit UnitLM model.

    Args:
        model_path: Path to model checkpoint
        tokenizer_name: HuggingFace tokenizer name
        device: Device to load on
        use_bf16: Whether to use bfloat16

    Returns:
        (model, tokenizer, text_vocab_size) tuple
    """
    print(f"Loading SlamKit UnitLM model from {model_path}...")

    # Load model using UnitLM
    model = UnitLM.from_pretrained(
        str(model_path),
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type == "cpu":
        model = model.to(device)
    elif use_bf16:
        model = model.to(torch.bfloat16)

    model.eval()

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )

    # Add special tokens if using ChatML format
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    # Get text vocab size from tokenizer
    text_vocab_size = len(tokenizer)

    print(f"Model loaded successfully!")
    print(f"Text vocabulary size: {text_vocab_size}")
    print(f"Total model vocabulary: {model.config.vocab_size}")

    return model, tokenizer


def predict_slamkit_model(model, tokenizer, prompt, device="cuda"):
    """
    Generate prediction from a SlamKit UnitLM model (text-only output).

    Args:
        model: SlamKit UnitLM model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt. Already formatted with render_mc.
        text_vocab_size: Size of text vocabulary
        device: Device to run on

    Returns:
        Generated text prediction
    """

    # Build ChatML format sequence
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Get tokenized text
    input_texts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # print("input_texts:", input_texts)

    # Tokenize
    input_ids = tokenizer.encode(input_texts, return_tensors="pt").to(device)

    # Get logits
    with torch.no_grad():
        logits = model(input_ids).logits

    # Compute the logits for the letters' ids
    letters = ['A', 'B', 'C', 'D']
    letter_to_id_cache = {}
    letter_ids = []
    for letter in letters:
        if not letter in letter_to_id_cache:
            encoded_letter = tokenizer.encode(letter)
            assert len(encoded_letter) == 1, "Each letter must be a single token"
            letter_to_id_cache[letter] = encoded_letter[0]
        letter_ids.append(letter_to_id_cache[letter])

    # Get logits for the last position
    focus_logits = logits[0, -1, letter_ids]
    predicted_answer = letters[torch.argmax(focus_logits).item()]
    return predicted_answer


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / f"log_{args.model_name}_{args.config}.txt"

    print(f"Loading model from {args.model_path}...")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Try to detect if this is a SlamKit model by checking config
    config_path = Path(args.model_path) / "config.json"
    is_slamkit_model = False

    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
            if config.get("model_type") == "speech_language_model" or "UnitLM" in config.get("architectures", []):
                is_slamkit_model = True

    if is_slamkit_model:
        print("Detected SlamKit UnitLM model")
        model, tokenizer = load_slamkit_model(
            args.model_path,
            args.tokenizer_name,
            device,
            args.bf16
        )
        predict_fn = lambda prompt: predict_slamkit_model(
            model, tokenizer, prompt, device
        )
    else:
        raise NotImplementedError("Only SlamKit UnitLM model loading is implemented in this script.")

    print("Model loaded successfully!")

    # Load dataset
    print(f"Loading dataset: edinburgh-dawg/mmlu-redux, config: {args.config}")
    dataset = load_dataset("edinburgh-dawg/mmlu-redux", args.config, split="test")
    print(f"Dataset size: {len(dataset)}")

    # Prepare results
    results = []
    correct = 0
    total = 0

    # Evaluate
    print("Starting evaluation...")
    N = 10 if args.quick else len(dataset)
    for i in tqdm(range(N)):
        question = dataset[i]["question"]
        choices = dataset[i]["choices"]
        answer_idx = dataset[i]["answer"]
        ground_truth = chr(65 + answer_idx)  # Convert 0,1,2,3 to A,B,C,D

        # Format question
        formatted_question = render_mc(question, ['A', 'B', 'C', 'D'], choices)

        # Get model prediction
        predicted_answer = predict_fn(formatted_question)

        # Check correctness
        is_correct = (predicted_answer == ground_truth) if predicted_answer else False
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "choices": str(choices),
            "ground_truth": ground_truth,
            "ground_truth_idx": answer_idx,
            "predicted_answer": predicted_answer if predicted_answer else "INVALID",
            "correct": is_correct,
            "corruptions": dataset[i].get("corruptions", "")
        })

    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"Results for {args.config}")
    print(f"{'='*50}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"{'='*50}")

    # Save results
    pred_df = pd.DataFrame(results)
    output_csv = output_dir / f"predictions_{args.model_name}_{args.config}.csv"
    pred_df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")

    # Save summary to log file
    with open(log_file, "w") as f:
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Model Name: {args.model_name}\n")
        f.write(f"Config: {args.config}\n")
        f.write(f"Total questions: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Accuracy: {accuracy:.2%}\n")

    print(f"Summary saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Zero-shot MMLU accuracy evaluation for SlamKit and HuggingFace models"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name for logging (default: derived from model_path)",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Tokenizer name for SlamKit models (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="abstract_algebra",
        help="MMLU-Redux subject configuration to evaluate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/mmlu_redux_eval",
        help="Directory to save results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick test with only 10 samples",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Use bfloat16 precision",
    )
    parser.add_argument(
        "--think",
        action="store_true",
        help="Enable thinking mode",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage (default: use CUDA if available)",
    )

    args = parser.parse_args()

    # Set model_name from path if not provided
    if args.model_name is None:
        args.model_name = Path(args.model_path).name

    main(args)
