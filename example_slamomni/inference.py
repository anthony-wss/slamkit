#!/usr/bin/env python3
"""
Interactive inference script for the trained SFT model.
Generates both text and speech (CosyVoice tokens) from user prompts.
"""

import torch
import json
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add parent dir to path
from transformers import AutoTokenizer
from slamkit.model import UnitLM


class SlamOmniInference:
    def __init__(
        self,
        model_path: str = "example_slamomni/output/checkpoint-125",
        tokenizer_name: str = "Qwen/Qwen3-0.6B",
        text_vocab_size: int = 151669,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the inference engine.

        Args:
            model_path: Path to the trained model checkpoint
            tokenizer_name: HuggingFace tokenizer name
            text_vocab_size: Size of text vocabulary (speech tokens start after this)
            device: Device to run inference on
        """
        print(f"Loading model from {model_path}...")
        self.device = device
        self.text_vocab_size = text_vocab_size

        # Load model using UnitLM (SlamKit's custom model class)
        model_path = Path(model_path).resolve()  # Convert to absolute path
        print(f"Loading SlamKit UnitLM model...")
        self.model = UnitLM.from_pretrained(
            str(model_path),
            device_map="auto",  # Don't use auto device mapping
        )
        self.model.to(device)
        if device == "cuda":
            self.model = self.model.half()  # Use bfloat16/fp16 on GPU
        self.model.eval()
        print(f"Model loaded on {device}")

        # Load tokenizer
        print(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )

        # Add special tokens
        special_tokens = ["<|im_start|>", "<|im_end|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

        print(f"Text vocabulary size: {self.text_vocab_size}")
        print(f"Total model vocabulary: {self.model.config.vocab_size}")
        print("Inference engine ready!\n")

    def generate_response(
        self,
        question: str,
        max_new_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ):
        """
        Generate text and speech response for a given question.

        Args:
            question: User's question
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling (False = greedy decoding)

        Returns:
            dict with 'text', 'speech_tokens', and 'full_response'
        """
        # Format prompt with ChatML
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        print(f"Generating response (max {max_new_tokens} tokens)...")

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                top_p=top_p if do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Extract generated tokens (remove input)
        generated_ids = output_ids[0][len(input_ids[0]):].cpu().tolist()

        # Separate text and speech tokens
        text_token_ids = []
        speech_token_ids = []

        for token_id in generated_ids:
            if token_id < self.text_vocab_size:
                text_token_ids.append(token_id)
            else:
                # Speech token (de-offset)
                speech_token_ids.append(token_id - self.text_vocab_size)

        # Decode text
        text_response = self.tokenizer.decode(text_token_ids, skip_special_tokens=False)

        # Clean up text (remove special tokens for display)
        text_display = text_response.replace("<|im_end|>", "").strip()

        return {
            "text": text_display,
            "speech_tokens": speech_token_ids,
            "full_response": self.tokenizer.decode(generated_ids, skip_special_tokens=False),
            "num_text_tokens": len(text_token_ids),
            "num_speech_tokens": len(speech_token_ids),
        }

    def save_speech_tokens(self, speech_tokens, output_path):
        """Save speech tokens to a file for later audio conversion."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON for easy loading
        with open(output_path, 'w') as f:
            json.dump({
                "speech_tokens": speech_tokens,
                "num_tokens": len(speech_tokens),
                "vocab_size": 8192,  # CosyVoice v2
                "frame_rate_hz": 25,
                "duration_seconds": len(speech_tokens) / 25,
            }, f, indent=2)

        print(f"Speech tokens saved to: {output_path}")
        return output_path


def interactive_mode():
    """Run interactive inference loop."""
    print("=" * 60)
    print("  SlamOmni Interactive Inference")
    print("=" * 60)
    print()

    # Initialize inference engine
    inference = SlamOmniInference()

    # Create output directory
    output_dir = Path("example_slamomni/generated")
    output_dir.mkdir(exist_ok=True)

    response_count = 0

    print("Type 'quit' or 'exit' to stop")
    print("=" * 60)
    print()

    while True:
        # Get user input
        question = input("\n🎤 Your question: ").strip()

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        if not question:
            continue

        print()
        print("-" * 60)

        # Generate response
        try:
            result = inference.generate_response(question)

            # Display results
            print(f"\n📝 Text Response:")
            print(f"{result['text']}")
            print()
            print(f"📊 Statistics:")
            print(f"  - Text tokens: {result['num_text_tokens']}")
            print(f"  - Speech tokens: {result['num_speech_tokens']}")
            print(f"  - Estimated audio duration: {result['num_speech_tokens'] / 25:.2f} seconds")

            # Save speech tokens
            response_count += 1
            tokens_path = output_dir / f"response_{response_count}_tokens.json"
            inference.save_speech_tokens(result['speech_tokens'], tokens_path)

        except Exception as e:
            print(f"\n❌ Error during generation: {e}")
            import traceback
            traceback.print_exc()

        print("-" * 60)


if __name__ == "__main__":
    interactive_mode()
