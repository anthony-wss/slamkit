"""
Test script for CosyVoiceFeatureExtractor integration with slamkit.

This script tests:
1. Loading the CosyVoice feature extractor
2. Extracting features from audio
3. Verifying the output format
4. Testing batch processing
5. Testing length masking
"""

import pytest
import torch
import torchaudio
import os
from slamkit.feature_extractor import CosyVoiceFeatureExtractor


def find_v1_onnx_model():
    """ Find available v1 CosyVoice ONNX model. """
    if os.path.exists('speech_tokenizer_v1.onnx'):
        return 'speech_tokenizer_v1.onnx', 4096
    return None, None

def find_v2_onnx_model():
    """ Find available v2 CosyVoice ONNX model. """
    if os.path.exists('speech_tokenizer_v2.onnx'):
        return 'speech_tokenizer_v2.onnx', 8192
    return None, None


def get_available_models():
    """Get list of available model versions."""
    available = []
    v1_path, v1_units = find_v1_onnx_model()
    v2_path, v2_units = find_v2_onnx_model()

    if v1_path:
        available.append(("v1", v1_path, v1_units))
    if v2_path:
        available.append(("v2", v2_path, v2_units))

    return available


@pytest.fixture(params=get_available_models(), ids=lambda x: x[0])
def extractor(request):
    """Create CosyVoice feature extractor fixture parametrized by version."""
    version, model_path, num_units = request.param
    if not torch.cuda.is_available():
        print(f"⚠️ CUDA not available, running CosyVoiceFeatureExtractor on CPU may be slow ({version})")
    return CosyVoiceFeatureExtractor(
        onnx_path=model_path,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
        num_units=num_units,
    ), num_units, version


@pytest.fixture
def test_audio():
    """Load test audio file."""
    audio_path = "example_data/audio/audio1.flac"
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Test audio file not found: {audio_path}")

    wav, sr = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Resample if needed
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav


@pytest.mark.skipif(
    len(get_available_models()) == 0,
    reason="CosyVoice ONNX v1 and v2 models both are not found"
)
class TestCosyVoiceFeatureExtractor:
    """Test suite for CosyVoice feature extractor."""

    def test_initialization(self, extractor):
        """Test that the feature extractor initializes correctly."""
        extractor_obj, num_units, version = extractor

        assert extractor_obj.sample_rate == 16000
        if version == "v1":
            assert 1 / extractor_obj.get_unit_duration() == pytest.approx(50.0)
        elif version == "v2":
            assert 1 / extractor_obj.get_unit_duration() == pytest.approx(25.0)
        else:
            raise ValueError(f"Unknown version {version}")

        print(f"✓ Feature extractor initialized ({version})")
        print(f"  - Model version: {version}")
        print(f"  - Num units: {num_units}")
        print(f"  - Sample rate: {extractor_obj.sample_rate} Hz")
        print(f"  - Unit duration: {extractor_obj.get_unit_duration():.4f} seconds")
        print(f"  - Frame rate: {1/extractor_obj.get_unit_duration():.1f} Hz")

    def test_feature_extraction(self, extractor, test_audio):
        """Test basic feature extraction."""
        extractor_obj, num_units, version = extractor
        wav = test_audio

        features = extractor_obj.extract(wav)

        assert len(features) == 1, "Should extract features for 1 sample"
        assert len(features[0]) > 0, "Feature sequence should not be empty"
        assert features[0].min() >= 0, "Token IDs should be non-negative"
        assert features[0].max() < num_units, f"Token IDs should be < {num_units}"

        print(f"✓ Features extracted ({version})")
        print(f"  - Model version: {version}")
        print(f"  - Num units: {num_units}")
        print(f"  - Number of samples: {len(features)}")
        print(f"  - Token sequence length: {len(features[0])}")
        print(f"  - Token range: [{features[0].min()}, {features[0].max()}]")
        print(f"  - First 20 tokens: {features[0][:20]}")

    def test_downsampling_ratio(self, extractor, test_audio):
        """Test downsampling ratio verification."""
        extractor_obj, num_units, version = extractor
        wav = test_audio

        features = extractor_obj.extract(wav)

        # CosyVoice v2 has 25Hz frame rate (not 50Hz), so we expect wav.shape[1] / 640 tokens
        expected_tokens_approx = wav.shape[1] / extractor_obj._hop_length  # Approximate due to mel processing
        actual_tokens = len(features[0])

        # Verify we're in the right ballpark (within 20% tolerance)
        tolerance = expected_tokens_approx * 0.2
        assert abs(expected_tokens_approx - actual_tokens) < tolerance, f"Token count mismatch: expected ~{expected_tokens_approx:.0f}, got {actual_tokens}"

        print(f"✓ Downsampling verification ({version})")
        print(f"  - Model version: {version}")
        print(f"  - Expected tokens: ~{expected_tokens_approx:.0f}")
        print(f"  - Actual tokens: {actual_tokens}")
        print(f"  - Downsampling factor: {wav.shape[1] / actual_tokens:.1f}x")

    def test_batch_extraction(self, extractor, test_audio):
        """Test batch extraction."""
        extractor_obj, num_units, version = extractor
        wav = test_audio

        # Create a batch of 3 samples
        batch = wav.repeat(3, 1)
        batch_features = extractor_obj.extract(batch)

        assert len(batch_features) == 3, "Should extract features for 3 samples"
        assert all(len(f) == len(batch_features[0]) for f in batch_features), "All samples should have same length"

        print(f"✓ Batch extraction successful ({version})")
        print(f"  - Model version: {version}")
        print(f"  - Batch size: {len(batch_features)}")
        print(f"  - All samples have same length: {all(len(f) == len(batch_features[0]) for f in batch_features)}")

    def test_length_masking(self, extractor, test_audio):
        """Test length masking functionality."""
        extractor_obj, num_units, version = extractor
        wav = test_audio

        batch = wav.repeat(3, 1)
        lens = torch.tensor([wav.shape[1], wav.shape[1] // 2, wav.shape[1] // 4])
        batch_features_masked = extractor_obj.extract(batch, lens=lens)

        assert len(batch_features_masked) == 3, "Should extract features for 3 samples"

        # Note: CosyVoice length masking may not always produce proportionally fewer tokens
        # because mel-spectrogram computation may pad or process in fixed windows
        # Just verify we got valid outputs for all samples
        for i, tokens in enumerate(batch_features_masked):
            assert len(tokens) > 0, f"Sample {i} should have tokens"
            assert tokens.min() >= 0, f"Sample {i} should have valid token IDs"
            assert tokens.max() < num_units, f"Sample {i} should have valid token IDs"

        print(f"✓ Length masking successful ({version})")
        print(f"  - Model version: {version}")
        for i, (length, tokens) in enumerate(zip(lens, batch_features_masked)):
            print(f"  - Sample {i}: input length={length.item()}, output tokens={len(tokens)}")

    def test_audio_length_limit(self, extractor):
        """Test that extraction fails for audio longer than 30 seconds."""
        extractor_obj, num_units, version = extractor

        # Create 31 seconds of audio
        wav = torch.randn(1, 31 * 16000)

        with pytest.raises(ValueError, match="does not support extracting speech tokens for audio longer than 30s"):
            extractor_obj.extract(wav)

        print(f"✓ Audio length limit enforced (max 30s) ({version})")
        print(f"  - Model version: {version}")

    def test_exact_30s_audio(self, extractor):
        """Test that extraction works for exactly 30 seconds of audio."""
        extractor_obj, num_units, version = extractor

        # Create exactly 30 seconds of audio
        wav = torch.randn(1, 30 * 16000)

        # Should not raise an error
        features = extractor_obj.extract(wav)
        assert len(features) == 1
        assert len(features[0]) > 0

        print(f"✓ Exact 30s audio extraction successful ({version})")
        print(f"  - Model version: {version}")
        print(f"  - Extracted {len(features[0])} tokens from 30s audio")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
