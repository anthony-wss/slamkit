#!/usr/bin/env python3
"""
Convert parquet data to SFT training format for text-only question -> text+speech response.
Format: <question_text><assistant_text_response><assistant_cosyvoice_token>
"""

import json
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path

def convert_to_sft_format(
    input_file: str,
    output_file: str,
    num_samples: int,
    tokenizer_name: str,
):
    """
    Convert parquet data to SFT training format.

    Args:
        input_file: Path to input parquet file
        output_file: Path to output JSONL file
        num_samples: Number of samples to process (default: 500)
        tokenizer_name: HuggingFace tokenizer to use for text encoding
    """
    print(f"Loading dataset from {input_file}...")
    dataset = load_dataset("parquet", data_files=input_file, split='train')

    # Limit to num_samples
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    print(f"Processing {len(dataset)} samples...")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Add special tokens if not present
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if added > 0:
        print(f"Added {added} special tokens to tokenizer")

    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Determine vocabulary size for text tokens
    text_vocab_size = len(tokenizer)
    print(f"Text vocabulary size: {text_vocab_size}")

    # CosyVoice tokenizer settings - need to offset speech tokens
    # Assuming CosyVoice uses 4096 units (v1 model)
    speech_vocab_size = 4096
    speech_offset = text_vocab_size

    print(f"Speech token offset: {speech_offset}")
    print(f"Total vocabulary size: {speech_offset + speech_vocab_size}")

    # Process samples
    output_samples = []
    for idx, sample in enumerate(dataset):
        question_text = sample['question']
        answer_text = sample['answer']
        answer_speech_tokens = sample['answer_cosyvoice_speech_token']

        # Build ChatML format sequence
        # Format: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer_text}\n{speech_tokens}<|im_end|>

        # User turn (question text only, no audio)
        user_part = f"<|im_start|>user\n{question_text}<|im_end|>\n"
        user_ids = tokenizer.encode(user_part, add_special_tokens=False)

        # Assistant turn prefix (text portion)
        assistant_prefix = f"<|im_start|>assistant\n{answer_text}\n"
        assistant_text_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False)

        # Assistant turn suffix (speech tokens - offset by text vocab size)
        assistant_speech_ids = [token + speech_offset for token in answer_speech_tokens]

        # Assistant turn end
        assistant_end_ids = [im_end_id]

        # Combine all parts
        input_ids = user_ids + assistant_text_ids + assistant_speech_ids + assistant_end_ids

        # Create labels: mask user portion (-100), keep assistant portion
        labels = (
            [-100] * len(user_ids) +  # Mask user turn
            assistant_text_ids +       # Keep assistant text
            assistant_speech_ids +     # Keep assistant speech
            assistant_end_ids          # Keep end token
        )

        # Attention mask (all 1s)
        attention_mask = [1] * len(input_ids)

        # Create output sample
        output_sample = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        output_samples.append(output_sample)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(dataset)} samples")

    # Write to output file
    print(f"Writing {len(output_samples)} samples to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for sample in output_samples:
            f.write(json.dumps(sample) + '\n')

    # Print statistics
    avg_length = sum(len(s['input_ids']) for s in output_samples) / len(output_samples)
    max_length = max(len(s['input_ids']) for s in output_samples)
    min_length = min(len(s['input_ids']) for s in output_samples)

    print(f"\n=== Statistics ===")
    print(f"Total samples: {len(output_samples)}")
    print(f"Average sequence length: {avg_length:.1f}")
    print(f"Max sequence length: {max_length}")
    print(f"Min sequence length: {min_length}")
    print(f"Text vocabulary size: {text_vocab_size}")
    print(f"Speech vocabulary size: {speech_vocab_size}")
    print(f"Total vocabulary size: {text_vocab_size + speech_vocab_size}")

    print(f"\nDone! Output written to: {output_file}")

    # Print first sample as example
    print(f"\n=== First sample (preview) ===")
    first_sample = output_samples[0]
    print(f"Input IDs length: {len(first_sample['input_ids'])}")
    print(f"Labels length: {len(first_sample['labels'])}")
    print(f"Attention mask length: {len(first_sample['attention_mask'])}")
    print(f"First 20 input IDs: {first_sample['input_ids'][:20]}")
    print(f"First 20 labels: {first_sample['labels'][:20]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert parquet data to SFT format")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/train-00000-of-00853.parquet",
        help="Input parquet file path"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./sft_data/train.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Number of samples to process"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace tokenizer name"
    )

    args = parser.parse_args()

    convert_to_sft_format(
        input_file=args.input_file,
        output_file=args.output_file,
        num_samples=args.num_samples,
        tokenizer_name=args.tokenizer,
    )
