#!/usr/bin/env python3
"""
Inspect a single sample from the converted SFT data.
"""

import json
from transformers import AutoTokenizer

# Load first sample
with open('sft_data/train.jsonl', 'r') as f:
    sample = json.loads(f.readline())

print("=== Sample Structure ===")
print(f"Keys: {list(sample.keys())}")
print(f"Input IDs length: {len(sample['input_ids'])}")
print(f"Labels length: {len(sample['labels'])}")
print(f"Attention mask length: {len(sample['attention_mask'])}")

# Load tokenizer to decode
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
tokenizer.add_special_tokens({"additional_special_tokens": ["<|im_start|>", "<|im_end|>"]})

print("\n=== Decoded Sequence ===")
# Decode input_ids (only text portion, not speech tokens)
text_vocab_size = 151665
text_tokens = [t for t in sample['input_ids'] if t < text_vocab_size]
decoded = tokenizer.decode(text_tokens, skip_special_tokens=False)
print(decoded)

print("\n=== Token Breakdown ===")
# Find where user ends and assistant begins
user_end_idx = None
for i, (inp_id, label_id) in enumerate(zip(sample['input_ids'], sample['labels'])):
    if label_id != -100:
        user_end_idx = i
        break

print(f"User portion length: {user_end_idx} tokens (masked with -100)")
print(f"Assistant portion length: {len(sample['input_ids']) - user_end_idx} tokens")

# Count speech tokens
speech_tokens = [t for t in sample['input_ids'] if t >= text_vocab_size]
print(f"Speech tokens count: {len(speech_tokens)}")
print(f"Speech token range: {min(speech_tokens) if speech_tokens else 'N/A'} - {max(speech_tokens) if speech_tokens else 'N/A'}")

# Show first few speech tokens (de-offset)
if speech_tokens:
    print(f"First 10 speech tokens (de-offset): {[t - text_vocab_size for t in speech_tokens[:10]]}")
