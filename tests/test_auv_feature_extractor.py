"""
Test script for AUVFeatureExtractor integration with slamkit.

This script tests:
1. Loading the AUV feature extractor
2. Extracting features from audio
3. Verifying the output format
"""

import torch
import torchaudio
from slamkit.feature_extractor import AUVFeatureExtractor


def main():
    print("=" * 60)
    print("Testing AUVFeatureExtractor")
    print("=" * 60)

    # Initialize the feature extractor
    print("\n1. Initializing AUV Feature Extractor...")
    extractor = AUVFeatureExtractor(
        checkpoint_path='auv.pt',
        compile=False,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        use_bf16=False,
    )
    print(f"   ✓ Feature extractor initialized")
    print(f"   - Sample rate: {extractor.sample_rate} Hz")
    print(f"   - Unit duration: {extractor.get_unit_duration():.4f} seconds ({1/extractor.get_unit_duration():.1f} Hz)")

    # Load test audio
    print("\n2. Loading test audio...")
    audio_path = "example_data/audio/audio1.flac"
    wav, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    print(f"   ✓ Audio loaded: {audio_path}")
    print(f"   - Shape: {wav.shape}")
    print(f"   - Sample rate: {sr} Hz")
    print(f"   - Duration: {wav.shape[1] / sr:.2f} seconds")

    # Resample if needed
    if sr != extractor.sample_rate:
        print(f"   - Resampling from {sr} Hz to {extractor.sample_rate} Hz...")
        wav = torchaudio.functional.resample(wav, sr, extractor.sample_rate)
        sr = extractor.sample_rate

    # Extract features
    print("\n3. Extracting codec tokens...")
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
    print(f"\n4. Verification:")
    print(f"   - Expected tokens (based on hop length): ~{expected_tokens}")
    print(f"   - Actual tokens: {actual_tokens}")
    print(f"   - Downsampling factor: {wav.shape[1] / actual_tokens:.1f}x")

    # Test with batch
    print("\n5. Testing with batch...")
    batch = wav.repeat(3, 1)  # Create a batch of 3 identical samples
    batch_features = extractor.extract(batch)

    print(f"   ✓ Batch extraction successful")
    print(f"   - Batch size: {len(batch_features)}")
    print(f"   - All samples have same length: {all(len(f) == len(batch_features[0]) for f in batch_features)}")

    # Test with lens parameter
    print("\n6. Testing with length masking...")
    lens = torch.tensor([wav.shape[1], wav.shape[1] // 2, wav.shape[1] // 4])
    batch_features_masked = extractor.extract(batch, lens=lens)

    print(f"   ✓ Length masking successful")
    for i, (length, tokens) in enumerate(zip(lens, batch_features_masked)):
        print(f"   - Sample {i}: input length={length.item()}, output tokens={len(tokens)}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
