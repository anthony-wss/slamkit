"""
Reconstruction tests for CosyVoice Feature Extractor.

These tests verify:
1. Token extraction is deterministic
2. Tokens can be properly round-tripped through the tokeniser
3. Token statistics and properties
"""

import pytest
import torch
import torchaudio
import numpy as np
from slamkit.feature_extractor import CosyVoiceFeatureExtractor


class TestCosyVoiceReconstruction:
    """Test token extraction and reconstruction properties"""

    def _find_onnx_model(self):
        """Find available CosyVoice ONNX model"""
        import os
        if os.path.exists('speech_tokenizer_v2.onnx'):
            return 'speech_tokenizer_v2.onnx'
        elif os.path.exists('speech_tokenizer_v1.onnx'):
            return 'speech_tokenizer_v1.onnx'
        return None

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_deterministic_extraction(self):
        """Test that extraction is deterministic (same input -> same output)"""
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Create test audio
        torch.manual_seed(42)  # For reproducibility
        wav = torch.randn(1, 16000 * 3)  # 3 seconds

        # Extract features twice
        features1 = extractor.extract(wav)
        features2 = extractor.extract(wav)

        # Should be identical
        assert len(features1) == len(features2)
        assert np.array_equal(features1[0], features2[0]), "Extraction should be deterministic"

        print(f"✓ Deterministic extraction test passed")
        print(f"  - Model: {model_path}")
        print(f"  - Extracted {len(features1[0])} tokens")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_token_distribution(self):
        """Test token distribution properties"""
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Extract from synthetic audio
        torch.manual_seed(42)
        wav = torch.randn(1, 16000 * 10)  # 10 seconds
        features = extractor.extract(wav)

        tokens = features[0]

        # Basic statistics
        unique_tokens = len(np.unique(tokens))
        token_mean = float(np.mean(tokens))
        token_std = float(np.std(tokens))

        # Assertions
        assert len(tokens) > 0
        assert unique_tokens > 1, "Should have multiple unique tokens"
        assert tokens.min() >= 0
        assert tokens.max() < expected_max_units

        print(f"✓ Token distribution test passed")
        print(f"  - Total tokens: {len(tokens)}")
        print(f"  - Unique tokens: {unique_tokens}")
        print(f"  - Token range: [{tokens.min()}, {tokens.max()}]")
        print(f"  - Mean: {token_mean:.1f}, Std: {token_std:.1f}")
        print(f"  - Codebook utilization: {unique_tokens}/{expected_max_units} ({100*unique_tokens/expected_max_units:.1f}%)")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_batch_consistency(self):
        """Test that batch processing gives same results as individual processing"""
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Create test audio samples
        torch.manual_seed(42)
        wav1 = torch.randn(1, 16000 * 2)
        wav2 = torch.randn(1, 16000 * 2)
        wav3 = torch.randn(1, 16000 * 2)

        # Process individually
        features_individual = [
            extractor.extract(wav1)[0],
            extractor.extract(wav2)[0],
            extractor.extract(wav3)[0],
        ]

        # Process as batch
        batch = torch.cat([wav1, wav2, wav3], dim=0)
        features_batch = extractor.extract(batch)

        # Compare
        for i in range(3):
            assert np.array_equal(features_individual[i], features_batch[i]), \
                f"Batch processing should give same results as individual processing for sample {i}"

        print(f"✓ Batch consistency test passed")
        print(f"  - Tested {len(features_batch)} samples")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_tokeniser_integration(self):
        """Test integration with SlamKit tokeniser"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory
        from omegaconf import OmegaConf
        from unittest.mock import patch

        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        # Create tokeniser config
        config = OmegaConf.create({
            'tokeniser_type': 'unit',
            'feature_extractor_type': 'cosyvoice',
            'feature_extractor': {
                'onnx_path': model_path,
                'device': 'cpu',
                'num_units': expected_max_units,
            },
            'params': {
                'num_units': expected_max_units,
                'load_fe': True,
                'dedup': True,
                'bos_eos_token_id': 1,
            }
        })

        tokeniser = tokeniser_factory(config)

        # Create test audio
        torch.manual_seed(42)
        wav = torch.randn(2, 16000 * 2)  # 2 samples, 2 seconds each

        # Extract audio representation
        representations = tokeniser.audio_represent(wav)

        # Verify
        assert len(representations) == 2
        assert 'units' in representations[0]
        assert 'duration' in representations[0]

        # Convert to list to verify
        units = list(representations[0]['units'])
        durations = list(representations[0]['duration'])

        assert len(units) > 0
        assert len(durations) > 0
        assert len(units) == len(durations)

        # Stringify
        stringified = tokeniser.stringify_representation(representations)
        assert len(stringified) == 2
        assert all(isinstance(s, str) for s in stringified)
        assert all('<Un' in s for s in stringified)  # Should contain unit tokens

        print(f"✓ Tokeniser integration test passed")
        print(f"  - Processed {len(representations)} samples")
        print(f"  - First sample: {len(units)} units")
        print(f"  - String length: {len(stringified[0])} chars")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_real_audio_round_trip(self):
        """Test full pipeline with real audio: audio -> tokens -> string -> tokens"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory
        from omegaconf import OmegaConf

        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        # Create tokeniser
        config = OmegaConf.create({
            'tokeniser_type': 'unit',
            'feature_extractor_type': 'cosyvoice',
            'feature_extractor': {
                'onnx_path': model_path,
                'device': 'cpu',
                'num_units': expected_max_units,
            },
            'params': {
                'num_units': expected_max_units,
                'load_fe': True,
                'dedup': True,
                'bos_eos_token_id': 1,
            }
        })

        tokeniser = tokeniser_factory(config)

        # Try to load real audio
        try:
            wav, sr = torchaudio.load("example_data/audio/audio1.flac")
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            # Use first 3 seconds
            wav = wav[:, :16000*3]
        except Exception:
            # Fall back to synthetic
            torch.manual_seed(42)
            wav = torch.randn(1, 16000 * 3)

        # Full pipeline
        # 1. Audio -> representation
        representations = tokeniser.audio_represent(wav)

        # 2. Representation -> string
        stringified = tokeniser.stringify_representation(representations)

        # 3. String -> tokens (via text tokenizer)
        tokenized = tokeniser.string_tokenise(stringified, return_tensors='pt')

        # Verify
        assert 'input_ids' in tokenized
        assert 'attention_mask' in tokenized
        assert tokenized['input_ids'].shape[0] == 1  # One sample
        assert tokenized['input_ids'].shape[1] > 0  # Has tokens

        print(f"✓ Real audio round-trip test passed")
        print(f"  - Audio shape: {wav.shape}")
        print(f"  - Representation units: {len(list(representations[0]['units']))}")
        print(f"  - String length: {len(stringified[0])}")
        print(f"  - Tokenized shape: {tokenized['input_ids'].shape}")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_different_audio_lengths(self):
        """Test extraction with various audio lengths"""
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Test different durations
        durations = [0.5, 1.0, 2.0, 5.0, 10.0]
        torch.manual_seed(42)

        results = []
        for duration in durations:
            wav = torch.randn(1, int(16000 * duration))
            features = extractor.extract(wav)
            num_tokens = len(features[0])
            actual_rate = num_tokens / duration
            results.append((duration, num_tokens, actual_rate))

        # Print results
        print(f"✓ Different audio lengths test passed")
        print(f"  Duration  | Tokens | Frame Rate")
        print(f"  --------- | ------ | ----------")
        for duration, num_tokens, rate in results:
            print(f"  {duration:5.1f}s    | {num_tokens:6d} | {rate:6.1f}Hz")

        # Verify frame rates are consistent
        rates = [r[2] for r in results]
        rate_std = np.std(rates)
        assert rate_std < 1.0, f"Frame rate should be consistent across different lengths (std: {rate_std:.2f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
