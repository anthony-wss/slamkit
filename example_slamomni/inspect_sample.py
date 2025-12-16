#!/usr/bin/env python3
"""
Inspect a single sample from the converted SFT data.
Verifies the new format with modality tokens: <text>, </text>, <speech>, </speech>
"""

import json
import argparse
from transformers import AutoTokenizer

def inspect_sample(file_path: str, sample_idx: int = 0):
    """
    Inspect a sample from the SFT data.

    Args:
        file_path: Path to the JSONL file
        sample_idx: Index of the sample to inspect (0-based)
    """
    # Load the specified sample
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                sample = json.loads(line)
                break
        else:
            raise ValueError(f"Sample index {sample_idx} not found in file")

    print("=== Sample Structure ===")
    print(f"Keys: {list(sample.keys())}")
    print(f"Input IDs length: {len(sample['input_ids'])}")
    print(f"Labels length: {len(sample['labels'])}")
    print(f"Attention mask length: {len(sample['attention_mask'])}")

    # Load tokenizer and add modality tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    modality_tokens = ["<text>", "</text>", "<speech>", "</speech>"]
    tokenizer.add_special_tokens({"additional_special_tokens": modality_tokens})

    # Get token IDs
    text_start_id = tokenizer.convert_tokens_to_ids("<text>")
    text_end_id = tokenizer.convert_tokens_to_ids("</text>")
    speech_start_id = tokenizer.convert_tokens_to_ids("<speech>")
    speech_end_id = tokenizer.convert_tokens_to_ids("</speech>")
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    print(f"\n=== Special Token IDs ===")
    print(f"<text>: {text_start_id}")
    print(f"</text>: {text_end_id}")
    print(f"<speech>: {speech_start_id}")
    print(f"</speech>: {speech_end_id}")
    print(f"<|im_start|>: {im_start_id}")
    print(f"<|im_end|>: {im_end_id}")

    # Determine vocabulary sizes
    text_vocab_size = len(tokenizer)
    speech_vocab_size = 4096  # CosyVoice v1
    speech_offset = text_vocab_size

    print(f"\n=== Vocabulary Info ===")
    print(f"Text vocabulary size: {text_vocab_size}")
    print(f"Speech vocabulary size: {speech_vocab_size}")
    print(f"Speech token offset: {speech_offset}")
    print(f"Total vocabulary size: {speech_offset + speech_vocab_size}")

    print("\n=== Decoded Sequence (Text Portion Only) ===")
    # Decode input_ids (only text portion, not speech tokens)
    text_tokens = [t for t in sample['input_ids'] if t < text_vocab_size]
    decoded = tokenizer.decode(text_tokens, skip_special_tokens=False)
    print(decoded)

    print("\n=== Token Breakdown ===")
    # Find where training starts (first non-masked label)
    training_start_idx = None
    for i, label_id in enumerate(sample['labels']):
        if label_id != -100:
            training_start_idx = i
            break

    print(f"Masked portion length: {training_start_idx} tokens (user part, masked with -100)")
    print(f"Training portion length: {len(sample['input_ids']) - training_start_idx} tokens (assistant response)")

    # Find modality token positions
    input_ids = sample['input_ids']
    text_start_pos = input_ids.index(text_start_id) if text_start_id in input_ids else None
    text_end_pos = input_ids.index(text_end_id) if text_end_id in input_ids else None
    speech_start_pos = input_ids.index(speech_start_id) if speech_start_id in input_ids else None
    speech_end_pos = input_ids.index(speech_end_id) if speech_end_id in input_ids else None

    print(f"\n=== Modality Token Positions ===")
    print(f"<text> position: {text_start_pos}")
    print(f"</text> position: {text_end_pos}")
    print(f"<speech> position: {speech_start_pos}")
    print(f"</speech> position: {speech_end_pos}")

    # Verify sequence structure
    print(f"\n=== Sequence Structure Verification ===")
    if text_start_pos is not None and text_end_pos is not None:
        text_portion_length = text_end_pos - text_start_pos - 1
        print(f"✓ Text portion length: {text_portion_length} tokens (between <text> and </text>)")
    else:
        print("✗ Missing text modality tokens!")

    if speech_start_pos is not None and speech_end_pos is not None:
        speech_portion_length = speech_end_pos - speech_start_pos - 1
        print(f"✓ Speech portion length: {speech_portion_length} tokens (between <speech> and </speech>)")
    else:
        print("✗ Missing speech modality tokens!")

    # Verify order
    if all([text_start_pos, text_end_pos, speech_start_pos, speech_end_pos]):
        if text_start_pos < text_end_pos < speech_start_pos < speech_end_pos:
            print("✓ Token order correct: <text> ... </text> <speech> ... </speech>")
        else:
            print("✗ Token order incorrect!")

    # Verify training starts at <text>
    if training_start_idx == text_start_pos:
        print(f"✓ Training starts at <text> token (position {training_start_idx})")
    else:
        print(f"✗ Training starts at position {training_start_idx}, but <text> is at {text_start_pos}")

    # Count speech tokens
    speech_tokens = [t for t in sample['input_ids'][speech_start_pos+1:speech_end_pos] if t >= text_vocab_size]
    print(f"\n=== Speech Tokens ===")
    print(f"Speech tokens count: {len(speech_tokens)}")
    if speech_tokens:
        print(f"Speech token range (raw): {min(speech_tokens)} - {max(speech_tokens)}")
        print(f"Speech token range (de-offset): {min(speech_tokens) - speech_offset} - {max(speech_tokens) - speech_offset}")
        print(f"First 10 speech tokens (de-offset): {[t - speech_offset for t in speech_tokens[:10]]}")

    # Show detailed token view around modality boundaries
    print(f"\n=== Detailed Token View (Around Modality Boundaries) ===")

    def show_tokens_around(position, context=5, label=""):
        print(f"\n{label} (position {position}, context ±{context}):")
        start = max(0, position - context)
        end = min(len(input_ids), position + context + 1)
        print("Pos  | Input ID  | Label     | Attn | Decoded")
        print("-" * 70)
        for i in range(start, end):
            inp = input_ids[i]
            lbl = sample['labels'][i]
            attn = sample['attention_mask'][i]

            # Decode token
            if inp < text_vocab_size:
                try:
                    decoded = tokenizer.decode([inp], skip_special_tokens=False)
                    decoded = decoded.replace('\n', '\\n').replace('\r', '\\r')[:30]
                except:
                    decoded = "<decode error>"
            else:
                decoded = f"<speech:{inp - speech_offset}>"

            lbl_str = str(lbl) if lbl != -100 else "-100"
            marker = " <--" if i == position else ""
            print(f"{i:4d} | {inp:9d} | {lbl_str:9s} | {attn:4d} | {decoded}{marker}")

    if text_start_pos is not None:
        show_tokens_around(text_start_pos, context=5, label="Around <text>")

    if text_end_pos is not None:
        show_tokens_around(text_end_pos, context=5, label="Around </text>")

    if speech_start_pos is not None:
        show_tokens_around(speech_start_pos, context=5, label="Around <speech>")

    if speech_end_pos is not None:
        show_tokens_around(speech_end_pos, context=5, label="Around </speech>")

    # Final validation
    print(f"\n=== Final Validation ===")
    checks = []
    checks.append(("Lengths match", len(input_ids) == len(sample['labels']) == len(sample['attention_mask'])))
    checks.append(("Ends with <|im_end|>", input_ids[-1] == im_end_id))
    checks.append(("Has <text> token", text_start_id in input_ids))
    checks.append(("Has </text> token", text_end_id in input_ids))
    checks.append(("Has <speech> token", speech_start_id in input_ids))
    checks.append(("Has </speech> token", speech_end_id in input_ids))
    checks.append(("Correct token order", text_start_pos < text_end_pos < speech_start_pos < speech_end_pos if all([text_start_pos, text_end_pos, speech_start_pos, speech_end_pos]) else False))
    checks.append(("Training starts at <text>", training_start_idx == text_start_pos))

    for check_name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"{status} {check_name}")

    all_passed = all(passed for _, passed in checks)
    print(f"\n{'='*70}")
    if all_passed:
        print("✓ All validation checks passed!")
    else:
        print("✗ Some validation checks failed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SFT training sample")
    parser.add_argument(
        "--file",
        type=str,
        default="sft_data/train_debug_100.jsonl",
        help="Path to JSONL file"
    )
    parser.add_argument(
        "--sample_idx",
        type=int,
        default=0,
        help="Sample index to inspect (0-based)"
    )

    args = parser.parse_args()
    inspect_sample(args.file, args.sample_idx)
