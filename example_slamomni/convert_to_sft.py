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
        input_file: Path to input parquet file or glob pattern (e.g., /data/*.parquet)
        output_file: Path to output JSONL file
        num_samples: Number of samples to process (None for all samples)
        tokenizer_name: HuggingFace tokenizer to use for text encoding
    """
    print(f"Loading dataset from {input_file}...")

    # Use streaming mode to avoid loading everything into memory/disk
    dataset = load_dataset("parquet", data_files=input_file, split='train', streaming=True)
    if num_samples is not None and num_samples > 0:
        print(f"Using streaming mode to process {num_samples} samples...")
        # Take only the first num_samples
        dataset = dataset.take(num_samples)
        total_to_process = num_samples
    else:
        print(f"Loading full dataset...")
        total_to_process = "ALL"  # len(dataset)

    print(f"Processing up to {total_to_process} samples...")

    # Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    assert tokenizer_name == "Qwen/Qwen3-0.6B", "This script is designed for Qwen3-0.6B tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    # Add modality special tokens
    modality_tokens = ["<text>", "</text>", "<speech>", "</speech>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": modality_tokens})
    print(f"Added {num_added} modality tokens: {modality_tokens}")

    # Get special token IDs
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    text_start_id = tokenizer.convert_tokens_to_ids("<text>")
    text_end_id = tokenizer.convert_tokens_to_ids("</text>")
    speech_start_id = tokenizer.convert_tokens_to_ids("<speech>")
    speech_end_id = tokenizer.convert_tokens_to_ids("</speech>")

    # Determine vocabulary size for text tokens
    text_vocab_size = len(tokenizer)
    print(f"Text vocabulary size: {text_vocab_size}")

    # CosyVoice tokenizer settings - need to offset speech tokens
    # Assuming CosyVoice uses 4096 units (v1 model)
    speech_vocab_size = 4096
    speech_offset = text_vocab_size

    print(f"Speech token offset: {speech_offset}")
    print(f"Total vocabulary size: {speech_offset + speech_vocab_size}")

    # Define batch processing function
    def process_batch(batch):
        """Process a batch of samples and convert to SFT format."""
        batch_size = len(batch['question'])

        # Initialize output lists
        input_ids_list = []
        labels_list = []
        attention_mask_list = []
        valid_indices = []

        for i in range(batch_size):
            question_text = batch['question'][i]
            answer_text = batch['answer'][i]
            answer_speech_tokens = batch['answer_cosyvoice_speech_token'][i]

            # Parse speech tokens (convert to list of integers)
            try:
                speech_tokens_raw = [int(t) for t in answer_speech_tokens]
                # Offset speech tokens to avoid collision with text vocabulary
                speech_tokens = [t + speech_offset for t in speech_tokens_raw]
            except (ValueError, AttributeError) as e:
                print(f"Warning: Failed to parse speech tokens for batch sample {i}: {e}")
                continue

            # Build ChatML format sequence
            messages = [
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": answer_text}
            ]

            # Get tokenized text
            text_tokens = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            if text_tokens[-1] == tokenizer.encode("\n", add_special_tokens=False)[0]:
                text_tokens = text_tokens[:-1]

            # Find where assistant response starts
            first_im_start_pos = text_tokens.index(im_start_id)
            second_im_start_pos = text_tokens.index(im_start_id, first_im_start_pos + 1)
            assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
            response_start_pos = second_im_start_pos + 1 + len(assistant_prefix)

            # Insert modality tokens and speech tokens
            # Note: apply_chat_template with enable_thinking=False already adds <think></think>
            # Original sequence: ... <|im_start|>assistant\n<think></think>{answer_text}<|im_end|>
            # Target sequence: ... <|im_start|>assistant\n<think></think><text>{answer_text}</text><speech>{speech_tokens}</speech><|im_end|>

            # Remove final <|im_end|> if present
            if text_tokens[-1] == im_end_id:
                text_tokens_without_end = text_tokens[:-1]
            else:
                text_tokens_without_end = text_tokens

            # Find where <think></think> ends (the answer_text starts after </think>\n\n)
            # The sequence is: <|im_start|>assistant\n<think></think>\n\n{answer_text}
            think_end_tokens = tokenizer.encode("</think>", add_special_tokens=False)

            # Find the position of </think>
            think_end_pos = None
            for i in range(response_start_pos, len(text_tokens_without_end)):
                # Check if we found </think> token sequence
                if (i + len(think_end_tokens) <= len(text_tokens_without_end) and
                    text_tokens_without_end[i:i+len(think_end_tokens)] == think_end_tokens):
                    think_end_pos = i + len(think_end_tokens)
                    break

            if think_end_pos is None:
                raise ValueError("Could not find </think> token in the sequence")

            # Skip the newlines after </think> to find where answer_text actually starts
            # Qwen tokenizer encodes "\n\n" as a single token (271)
            newline_token = tokenizer.encode("\n\n", add_special_tokens=False)[0]
            if (think_end_pos < len(text_tokens_without_end) and
                text_tokens_without_end[think_end_pos] == newline_token):
                answer_text_start = think_end_pos + 1
            else:
                answer_text_start = think_end_pos

            answer_text_end = len(text_tokens_without_end)

            # Build the new sequence:
            # 1. User part + "assistant\n" + <think></think>\n\n
            # 2. <text>
            # 3. Answer text tokens
            # 4. </text>
            # 5. <speech>
            # 6. Speech tokens
            # 7. </speech>
            # 8. <|im_end|>
            input_ids = (
                text_tokens_without_end[:answer_text_start] +  # User part + "assistant\n<think></think>\n\n"
                [text_start_id] +  # <text>
                text_tokens_without_end[answer_text_start:answer_text_end] +  # Answer text
                [text_end_id] +  # </text>
                [speech_start_id] +  # <speech>
                speech_tokens +  # Speech tokens
                [speech_end_id] +  # </speech>
                [im_end_id]  # <|im_end|>
            )

            # Create labels: mask everything except assistant response
            # Assistant response now starts after "assistant\n" and includes:
            # <think></think><text>{answer_text}</text><speech>{speech_tokens}</speech><|im_end|>
            # We need to calculate the new response_start_pos after inserting tokens
            new_response_start_pos = input_ids.index(text_start_id)

            labels = (
                [-100] * new_response_start_pos +  # Mask user part + <think></think> + <text>
                input_ids[new_response_start_pos:]  # Train on answer text + </text> + <speech> + speech tokens + </speech> + <|im_end|>
            )

            # Attention mask
            attention_mask = [1] * len(input_ids)

            # Validation
            assert len(input_ids) == len(labels) == len(attention_mask), \
                f"Length mismatch: input_ids={len(input_ids)}, labels={len(labels)}, attention_mask={len(attention_mask)}"
            assert input_ids[-1] == im_end_id, f"Missing EOS token at end of sequence"
            assert labels[-1] == im_end_id, f"EOS token should not be masked in labels"

            input_ids_list.append(input_ids)
            labels_list.append(labels)
            attention_mask_list.append(attention_mask)
            valid_indices.append(i)

        return {
            "input_ids": input_ids_list,
            "labels": labels_list,
            "attention_mask": attention_mask_list,
        }

    # Apply batch processing using .map()
    print(f"Processing samples with batched .map()...")
    processed_dataset = dataset.map(
        process_batch,
        batched=True,
        batch_size=100,  # Process 100 samples at a time
        remove_columns=dataset.column_names if hasattr(dataset, 'column_names') else None,
    )

    # Write to output file incrementally and track statistics
    print(f"Writing to {output_file}...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    # Statistics tracking
    num_samples = 0
    total_length = 0
    min_length = float('inf')
    max_length = 0
    first_sample = None

    with open(output_file, 'w') as f:
        for idx, sample in enumerate(processed_dataset):
            # Write sample immediately
            output_sample = {
                "input_ids": sample['input_ids'],
                "labels": sample['labels'],
                "attention_mask": sample['attention_mask'],
            }
            f.write(json.dumps(output_sample) + '\n')

            # Update statistics
            seq_length = len(sample['input_ids'])
            num_samples += 1
            total_length += seq_length
            min_length = min(min_length, seq_length)
            max_length = max(max_length, seq_length)

            # Save first sample for preview
            if first_sample is None:
                first_sample = output_sample

            if (idx + 1) % 100 == 0:
                print(f"Processed and written {idx + 1} samples")

    # Print statistics
    avg_length = total_length / num_samples if num_samples > 0 else 0

    print(f"\n=== Statistics ===")
    print(f"Total samples: {num_samples}")
    print(f"Average sequence length: {avg_length:.1f}")
    print(f"Max sequence length: {max_length}")
    print(f"Min sequence length: {min_length if min_length != float('inf') else 0}")
    print(f"Text vocabulary size: {text_vocab_size}")
    print(f"Speech vocabulary size: {speech_vocab_size}")
    print(f"Total vocabulary size: {text_vocab_size + speech_vocab_size}")

    print(f"\nDone! Output written to: {output_file}")

    # Print first sample as example
    if first_sample is not None:
        print(f"\n=== First sample (preview) ===")
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
    else:
        print("\nNo samples processed - unable to show preview.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert parquet data to SFT format")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/data/*.parquet",
        help="Input parquet file path or glob pattern (e.g., /data/*.parquet)"
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
        default=None,
        help="Number of samples to process (default: None, process all)"
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
