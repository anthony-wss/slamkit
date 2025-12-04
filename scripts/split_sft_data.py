#!/usr/bin/env python3
"""
Split SFT data into train and eval sets with a reproducible random split.
Memory-efficient implementation for large files.

Usage:
    python cli/split_sft_data.py --input example_slamomni/sft_data/train_all.jsonl \
                                  --train_output example_slamomni/sft_data/train_v1.jsonl \
                                  --eval_output example_slamomni/sft_data/eval_v1.jsonl \
                                  --eval_ratio 0.15 \
                                  --seed 42
"""

import argparse
import random
from pathlib import Path


def count_lines(file_path: str) -> int:
    """Count lines in a file efficiently."""
    print(f"Counting lines in {file_path}...")
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def split_jsonl(input_path: str, train_output: str, eval_output: str, eval_ratio: float = 0.15, seed: int = 42):
    """
    Split a JSONL file into train and eval sets using a memory-efficient approach.

    Args:
        input_path: Path to input JSONL file
        train_output: Path to output train JSONL file
        eval_output: Path to output eval JSONL file
        eval_ratio: Ratio of data to use for evaluation (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Count total lines
    total_samples = count_lines(input_path)
    print(f"Total samples: {total_samples}")

    # Calculate split sizes
    eval_size = int(total_samples * eval_ratio)
    train_size = total_samples - eval_size

    print(f"Train samples: {train_size} ({100 * (1 - eval_ratio):.1f}%)")
    print(f"Eval samples: {eval_size} ({100 * eval_ratio:.1f}%)")

    # Create shuffled indices and convert eval indices to a set for O(1) lookup
    indices = list(range(total_samples))
    random.shuffle(indices)
    eval_indices = set(indices[:eval_size])

    # Create output directories if they don't exist
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(eval_output).parent.mkdir(parents=True, exist_ok=True)

    # Single pass through the file, writing to appropriate output
    print(f"Splitting data...")
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(train_output, 'w', encoding='utf-8') as f_train, \
         open(eval_output, 'w', encoding='utf-8') as f_eval:

        for idx, line in enumerate(f_in):
            if idx in eval_indices:
                f_eval.write(line)
            else:
                f_train.write(line)

            # Progress indicator
            if (idx + 1) % 50000 == 0:
                print(f"  Processed {idx + 1}/{total_samples} lines...")

    print(f"Written {train_output} ({train_size} samples)")
    print(f"Written {eval_output} ({eval_size} samples)")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Split SFT data into train and eval sets (memory-efficient)")
    parser.add_argument('--input', type=str, default='example_slamomni/sft_data/train_all.jsonl',
                        help='Path to input JSONL file')
    parser.add_argument('--train_output', type=str, default='example_slamomni/sft_data/train_v1.jsonl',
                        help='Path to output train JSONL file')
    parser.add_argument('--eval_output', type=str, default='example_slamomni/sft_data/eval_v1.jsonl',
                        help='Path to output eval JSONL file')
    parser.add_argument('--eval_ratio', type=float, default=0.15,
                        help='Ratio of data to use for evaluation (default: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    split_jsonl(
        input_path=args.input,
        train_output=args.train_output,
        eval_output=args.eval_output,
        eval_ratio=args.eval_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
