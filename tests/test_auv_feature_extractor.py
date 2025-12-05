"""
Test script for AUVFeatureExtractor integration with slamkit.

This script tests:
1. Loading the AUV feature extractor
2. Extracting features from audio
3. Verifying the output format
"""

import pytest
import torch
import torchaudio
from slamkit.feature_extractor import AUVFeatureExtractor


@pytest.fixture
def extractor():
    """Create AUV feature extractor fixture."""
    return AUVFeatureExtractor(
        checkpoint_path='auv.pt',
        compile=False,  # TODO: Test if compile can speed up feature extraction.
                        # If so, create separate tests for compiled version.
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        use_bf16=False,
    )


@pytest.fixture
def test_audio(extractor):
    """Load and prepare test audio."""
    audio_path = "example_data/audio/audio1.flac"
    wav, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if needed
    if sr != extractor.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, extractor.sample_rate)

    return wav


@pytest.mark.skipif(
    not __import__('os').path.exists('auv.pt'),
    reason="AUV checkpoint not found (auv.pt)"
)
class TestAUVFeatureExtractor:
    """Test suite for AUV feature extractor."""

    def test_initialization(self, extractor):
        """Test that the feature extractor initializes correctly."""
        assert extractor.sample_rate == 16000
        assert extractor.get_unit_duration() == pytest.approx(0.02)
        assert 1 / extractor.get_unit_duration() == pytest.approx(50.0)
        print(f"✓ Feature extractor initialized")
        print(f"  - Sample rate: {extractor.sample_rate} Hz")
        print(f"  - Unit duration: {extractor.get_unit_duration():.4f} seconds ({1/extractor.get_unit_duration():.1f} Hz)")

    def test_feature_extraction(self, extractor, test_audio):
        """Test basic feature extraction."""
        wav = test_audio
        features = extractor.extract(wav)

        assert len(features) == 1
        assert len(features[0]) > 0
        assert features[0].min() >= 0
        assert features[0].max() < 20480  # AUV has 20480 units

        print(f"✓ Features extracted")
        print(f"  - Number of samples: {len(features)}")
        print(f"  - Token sequence length: {len(features[0])}")
        print(f"  - Token range: [{features[0].min()}, {features[0].max()}]")
        print(f"  - First 20 tokens: {features[0][:20]}")

    def test_batch_extraction(self, extractor, test_audio):
        """Test batch extraction."""
        wav = test_audio
        batch = wav.repeat(3, 1)  # Create a batch of 3 identical samples
        batch_features = extractor.extract(batch)

        assert len(batch_features) == 3
        assert all(len(f) == len(batch_features[0]) for f in batch_features)

        print(f"✓ Batch extraction successful")
        print(f"  - Batch size: {len(batch_features)}")
        print(f"  - All samples have same length: {all(len(f) == len(batch_features[0]) for f in batch_features)}")

    def test_length_masking(self, extractor, test_audio):
        """Test length masking functionality."""
        wav = test_audio
        batch = wav.repeat(3, 1)
        lens = torch.tensor([wav.shape[1], wav.shape[1] // 2, wav.shape[1] // 4])
        batch_features_masked = extractor.extract(batch, lens=lens)

        assert len(batch_features_masked) == 3
        # Features should be shorter for shorter inputs
        assert len(batch_features_masked[1]) < len(batch_features_masked[0])
        assert len(batch_features_masked[2]) < len(batch_features_masked[1])

        print(f"✓ Length masking successful")
        for i, (length, tokens) in enumerate(zip(lens, batch_features_masked)):
            print(f"  - Sample {i}: input length={length.item()}, output tokens={len(tokens)}")
    

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="No GPU available"
    )
    def test_gpu_extraction(self, extractor, test_audio):
        """Test extraction on GPU if available."""
        wav = test_audio.to('cuda:0')
        features_gpu = extractor.extract(wav)

        assert len(features_gpu) == 1
        assert len(features_gpu[0]) > 0

        print(f"✓ GPU feature extraction successful")
        print(f"  - Number of samples: {len(features_gpu)}")
        print(f"  - Token sequence length: {len(features_gpu[0])}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
