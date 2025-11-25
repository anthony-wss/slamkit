"""
Prepare text-only SFT tokens for ARC-Easy training.

This script tokenizes text-only conversations (no audio) using ChatML formatting
for supervised fine-tuning.

Usage:
    python example_arc/prepare_arc_tokens.py \
        --data_path example_arc/arc_easy_train.jsonl \
        --out_path example_arc/arc_easy_train_tokens.jsonl \
        --tokenizer Qwen/Qwen3-0.6B
"""

import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer
from tqdm import tqdm


def prepare_text_only_tokens(data_path: str, out_path: str, tokenizer_path: str = "Qwen/Qwen3-0.6B"):
    """
    Tokenize text-only conversations for SFT training with ChatML formatting.

    Creates input_ids, labels, and attention_mask for each conversation.
    Labels mask out the user portion (-100) and only train on assistant responses.
    """
    print("=" * 60)
    print("Preparing Text-Only SFT Tokens")
    print("=" * 60)

    # Load tokenizer
    print(f"\nLoading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Add special tokens for ChatML format
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
    print(f"✓ Added {num_added} special tokens: {special_tokens}")

    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Get newline token ID
    newline_tokens = tokenizer.encode("\n", add_special_tokens=False)
    newline_id = newline_tokens[0] if newline_tokens else tokenizer.convert_tokens_to_ids("<0x0A>")

    print(f"✓ Special token IDs: <|im_start|>={im_start_id}, <|im_end|>={im_end_id}, \\n={newline_id}")

    # Load data
    print(f"\nLoading data from {data_path}")
    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]
    print(f"✓ Loaded {len(data)} conversations")

    # Process and tokenize
    print(f"\nTokenizing conversations...")
    output_data = []

    for sample in tqdm(data, desc="Processing"):
        user_text = sample['user_text']
        assistant_text = sample['assistant_text']

        # Tokenize text content
        user_text_tokens = tokenizer(user_text, add_special_tokens=False)['input_ids']
        assistant_text_tokens = tokenizer(assistant_text, add_special_tokens=False)['input_ids']

        # Build ChatML sequence:
        # <|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{assistant_text}<|im_end|>
        input_ids = (
            [im_start_id] +
            tokenizer("user", add_special_tokens=False)['input_ids'] +
            [newline_id] +
            user_text_tokens +
            [im_end_id, newline_id] +
            [im_start_id] +
            tokenizer("assistant", add_special_tokens=False)['input_ids'] +
            [newline_id] +
            assistant_text_tokens +
            [im_end_id]
        )

        # Create labels: mask user portion with -100
        # User portion: <|im_start|>user\n{user_text}<|im_end|>\n
        user_portion_length = (
            1 +  # <|im_start|>
            len(tokenizer("user", add_special_tokens=False)['input_ids']) +
            1 +  # \n
            len(user_text_tokens) +
            1 +  # <|im_end|>
            1    # \n
        )

        labels = [-100] * user_portion_length + input_ids[user_portion_length:]

        # Create attention mask (all 1s for now, no padding)
        attention_mask = [1] * len(input_ids)

        # Verify lengths match
        assert len(input_ids) == len(labels) == len(attention_mask), \
            f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}, attention_mask={len(attention_mask)}"

        # Store processed sample
        output_sample = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
        output_data.append(output_sample)

    # Create output directory if needed
    out_dir = Path(out_path).parent
    if out_dir and not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Write output
    print(f"\nWriting tokenized data to {out_path}")
    with open(out_path, 'w') as f:
        for sample in output_data:
            f.write(json.dumps(sample) + '\n')

    print(f"✓ Processed {len(output_data)} samples")

    # Show statistics
    total_tokens = sum(len(s['input_ids']) for s in output_data)
    masked_tokens = sum(sum(1 for l in s['labels'] if l == -100) for s in output_data)
    training_tokens = total_tokens - masked_tokens

    print(f"\nStatistics:")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Masked tokens (user): {masked_tokens:,}")
    print(f"  - Training tokens (assistant): {training_tokens:,}")
    print(f"  - Avg tokens per sample: {total_tokens / len(output_data):.1f}")

    print("\n" + "=" * 60)
    print("✓ Tokenization complete!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare text-only SFT tokens")
    parser.add_argument("--data_path", type=str, required=True, help="Input jsonl file with user_text and assistant_text")
    parser.add_argument("--out_path", type=str, required=True, help="Output jsonl file for tokenized data")
    parser.add_argument("--tokenizer", type=str, default="Qwen/Qwen3-0.6B", help="Tokenizer to use")
    args = parser.parse_args()

    prepare_text_only_tokens(args.data_path, args.out_path, args.tokenizer)
