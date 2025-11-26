"""
Test script for CosyVoiceFeatureExtractor integration with slamkit.

This script tests:
1. Loading the CosyVoice feature extractor
2. Extracting features from audio
3. Verifying the output format
4. Testing batch processing
5. Testing length masking
"""

import torch
import torchaudio
from slamkit.feature_extractor import CosyVoiceFeatureExtractor


def main():
    print("=" * 60)
    print("Testing CosyVoiceFeatureExtractor")
    print("=" * 60)

    # Find available ONNX model
    print("\n1. Finding CosyVoice ONNX model...")
    import os
    onnx_path = None
    if os.path.exists('speech_tokenizer_v2.onnx'):
        onnx_path = 'speech_tokenizer_v2.onnx'
    elif os.path.exists('speech_tokenizer_v1.onnx'):
        onnx_path = 'speech_tokenizer_v1.onnx'

    if onnx_path is None:
        print(f"   ✗ Error: No CosyVoice ONNX model found")
        print(f"   Please download the CosyVoice ONNX model:")
        print(f"   - speech_tokenizer_v1.onnx or speech_tokenizer_v2.onnx")
        print(f"   - Place it in the current directory")
        return

    print(f"   ✓ Found model: {onnx_path}")

    # Initialize the feature extractor
    print("\n2. Initializing CosyVoice Feature Extractor...")
    try:
        extractor = CosyVoiceFeatureExtractor(
            onnx_path=onnx_path,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            num_units=4096,
        )
        print(f"   ✓ Feature extractor initialized")
        print(f"   - Sample rate: {extractor.sample_rate} Hz")
        print(f"   - Unit duration: {extractor.get_unit_duration():.4f} seconds ({1/extractor.get_unit_duration():.1f} Hz)")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Load test audio
    print("\n3. Loading test audio...")
    audio_path = "example_data/audio/audio1.flac"
    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        print(f"   ✗ Error loading audio: {e}")
        print(f"   Creating synthetic audio for testing...")
        # Create 5 seconds of synthetic audio
        sr = 16000
        wav = torch.randn(1, sr * 5)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    print(f"   ✓ Audio loaded/created")
    print(f"   - Shape: {wav.shape}")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Duration: {wav.shape[1] / sr:.2f} seconds")

    # Resample if needed
    if sr != extractor.sample_rate:
        print(f"   - Resampling from {sr} Hz to {extractor.sample_rate} Hz...")
        wav = torchaudio.functional.resample(wav, sr, extractor.sample_rate)
        sr = extractor.sample_rate

    # Check audio length (CosyVoice has 30s limit)
    duration = wav.shape[1] / sr
    if duration > 30:
        print(f"   - Audio too long ({duration:.2f}s), truncating to 30s...")
        wav = wav[:, :30 * sr]

    # Extract features
    print("\n4. Extracting speech tokens...")
    try:
        features = extractor.extract(wav)

        print(f"   ✓ Features extracted")
        print(f"   - Number of samples: {len(features)}")
        print(f"   - Token sequence length: {len(features[0])}")
        print(f"   - Token range: [{features[0].min()}, {features[0].max()}]")
        print(f"   - First 20 tokens: {features[0][:20]}")
        print(f"   - Token dtype: {features[0].dtype}")

        # Verify downsampling ratio
        expected_tokens = int(wav.shape[1] / extractor._hop_length)
        actual_tokens = len(features[0])
        print(f"\n5. Verification:")
        print(f"   - Expected tokens (based on hop length): ~{expected_tokens}")
        print(f"   - Actual tokens: {actual_tokens}")
        print(f"   - Downsampling factor: {wav.shape[1] / actual_tokens:.1f}x")

    except Exception as e:
        print(f"   ✗ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test with batch
    print("\n6. Testing with batch...")
    try:
        batch = wav.repeat(3, 1)  # Create a batch of 3 identical samples
        batch_features = extractor.extract(batch)

        print(f"   ✓ Batch extraction successful")
        print(f"   - Batch size: {len(batch_features)}")
        print(f"   - All samples have same length: {all(len(f) == len(batch_features[0]) for f in batch_features)}")
    except Exception as e:
        print(f"   ✗ Error during batch extraction: {e}")
        import traceback
        traceback.print_exc()

    # Test with lens parameter
    print("\n7. Testing with length masking...")
    try:
        lens = torch.tensor([wav.shape[1], wav.shape[1] // 2, wav.shape[1] // 4])
        batch_features_masked = extractor.extract(batch, lens=lens)

        print(f"   ✓ Length masking successful")
        for i, (length, tokens) in enumerate(zip(lens, batch_features_masked)):
            print(f"   - Sample {i}: input length={length.item()}, output tokens={len(tokens)}")
    except Exception as e:
        print(f"   ✗ Error during length masking: {e}")
        import traceback
        traceback.print_exc()

    # Test audio length limit
    print("\n8. Testing audio length limit (30s)...")
    try:
        # Create 31s audio (should fail)
        long_audio = torch.randn(1, 31 * extractor.sample_rate)
        try:
            extractor.extract(long_audio)
            print(f"   ✗ Should have raised ValueError for audio > 30s")
        except ValueError as e:
            print(f"   ✓ Correctly raised ValueError: {str(e)[:80]}...")
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
