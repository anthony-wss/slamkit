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
    assert tokenizer_name == "Qwen/Qwen3-0.6B", "This script is designed for Qwen3-0.6B tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

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

        # Parse speech tokens (convert string to list of integers)
        # Format: "1234 5678 9012" -> [1234, 5678, 9012]
        try:
            speech_tokens_raw = [int(t) for t in answer_speech_tokens]
            # Offset speech tokens to avoid collision with text vocabulary
            speech_tokens = [t + speech_offset for t in speech_tokens_raw]
        except (ValueError, AttributeError) as e:
            print(f"Warning: Failed to parse speech tokens for sample {idx}: {e}")
            print(f"Speech tokens: {answer_speech_tokens}")
            continue

        # Build ChatML format sequence with TEXT ONLY first
        # We'll manually append speech tokens after
        # Format: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer_text}
        messages = [
            {"role": "user", "content": question_text},
            {"role": "assistant", "content": answer_text}
        ]

        # Get tokenized text (this includes EOS at the end)
        text_tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        if text_tokens[-1] == tokenizer.encode("\n", add_special_tokens=False)[0]:
            # Remove trailing newline token if present
            text_tokens = text_tokens[:-1]

        # Find where assistant response actually starts
        # Structure: <|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer_text}<|im_end|>
        # We want to find the position right after "<|im_start|>assistant\n"

        # Find the second <|im_start|> (the assistant one)
        first_im_start_pos = text_tokens.index(im_start_id)
        second_im_start_pos = text_tokens.index(im_start_id, first_im_start_pos + 1)

        # Tokenize "assistant\n" to find its length
        assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
        # Position where actual response content starts (after <|im_start|>assistant\n)
        response_start_pos = second_im_start_pos + 1 + len(assistant_prefix)

        # Now we need to insert speech tokens BEFORE the final <|im_end|>
        # Remove the final <|im_end|> temporarily
        if text_tokens[-1] == im_end_id:
            text_tokens_without_end = text_tokens[:-1]
            # Append speech tokens and then <|im_end|>
            input_ids = text_tokens_without_end + speech_tokens + [im_end_id]
        else:
            # No <|im_end|> found - this shouldn't happen with apply_chat_template
            print(f"Warning: No <|im_end|> found at end of sample {idx}")
            input_ids = text_tokens + speech_tokens + [im_end_id]

        # Create labels: mask everything except the assistant response content + speech + EOS
        # Structure: [-100, -100, ..., -100, actual_response_tokens, speech_tokens, im_end_id]
        labels = (
            [-100] * response_start_pos +  # Mask user + assistant prefix
            input_ids[response_start_pos:]  # Keep assistant response + speech + EOS
        )

        # Attention mask (all 1s - attend to everything)
        attention_mask = [1] * len(input_ids)

        # Validation checks
        assert len(input_ids) == len(labels) == len(attention_mask), \
            f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}, attention_mask={len(attention_mask)}"
        assert input_ids[-1] == im_end_id, f"Missing EOS token at end of sequence"
        assert labels[-1] == im_end_id, f"EOS token should not be masked in labels"

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

    # Show the structure visually
    print("\n--- Token Structure (first 30 positions) ---")
    print("Pos | Input ID | Label    | Attn | Decoded Token")
    print("-" * 70)
    for i in range(min(30, len(first_sample['input_ids']))):
        inp = first_sample['input_ids'][i]
        lbl = first_sample['labels'][i]
        attn = first_sample['attention_mask'][i]

        # Try to decode the token (only if it's in text vocab range)
        if inp < text_vocab_size:
            try:
                decoded = tokenizer.decode([inp], skip_special_tokens=False)
                decoded = decoded.replace('\n', '\\n')[:20]  # Truncate long tokens
            except:
                decoded = "<decode error>"
        else:
            decoded = f"<speech:{inp - speech_offset}>"

        # Show if label is masked
        lbl_str = str(lbl) if lbl != -100 else "-100 (MASK)"

        print(f"{i:3d} | {inp:8d} | {lbl_str:8s} | {attn:4d} | {decoded}")

    print("\n--- Last 10 positions (should end with EOS) ---")
    start_idx = max(0, len(first_sample['input_ids']) - 10)
    print("Pos | Input ID | Label    | Attn | Decoded Token")
    print("-" * 70)
    for i in range(start_idx, len(first_sample['input_ids'])):
        inp = first_sample['input_ids'][i]
        lbl = first_sample['labels'][i]
        attn = first_sample['attention_mask'][i]

        if inp < text_vocab_size:
            try:
                decoded = tokenizer.decode([inp], skip_special_tokens=False)
                decoded = decoded.replace('\n', '\\n')[:20]
            except:
                decoded = "<decode error>"
        else:
            decoded = f"<speech:{inp - speech_offset}>"

        lbl_str = str(lbl) if lbl != -100 else "-100 (MASK)"
        print(f"{i:3d} | {inp:8d} | {lbl_str:8s} | {attn:4d} | {decoded}")

    # Verify correctness
    print("\n=== Validation ===")
    num_masked = sum(1 for l in first_sample['labels'] if l == -100)
    num_trainable = sum(1 for l in first_sample['labels'] if l != -100)
    print(f"Masked tokens (no loss): {num_masked}")
    print(f"Trainable tokens (with loss): {num_trainable}")
    print(f"Last token is EOS: {first_sample['input_ids'][-1] == im_end_id}")
    print(f"EOS is in labels (not masked): {first_sample['labels'][-1] == im_end_id}")


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
