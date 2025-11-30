#!/usr/bin/env python3
"""
Test script for inference - runs a single example
"""

from inference import SlamOmniInference
from pathlib import Path

def test_inference():
    print("Initializing inference engine...")
    inference = SlamOmniInference(
        model_path="sft_overfit_model/checkpoint-600"  # Relative to example_slamomni/
    )

    # Test question
    # question = "What is artificial intelligence?"
    question = "What makes you different from other assistant models?"

    print(f"\nTest question: {question}")
    print("="  * 60)

    # Generate response
    result = inference.generate_response(
        question,
        max_new_tokens=500,
        do_sample=False,  # Greedy decoding for reproducibility
    )

    # Display results
    print(f"\n📝 Text Response:")
    print(f"{result['text']}")
    print()
    print(f"📊 Statistics:")
    print(f"  - Text tokens: {result['num_text_tokens']}")
    print(f"  - Speech tokens: {result['num_speech_tokens']}")
    print(f"  - Estimated audio duration: {result['num_speech_tokens'] / 25:.2f} seconds")
    print()

    # Save tokens
    output_dir = Path("./generated")
    output_dir.mkdir(exist_ok=True)
    tokens_path = output_dir / "test_response_tokens.json"
    inference.save_speech_tokens(result['speech_tokens'], tokens_path)

if __name__ == "__main__":
    test_inference()
