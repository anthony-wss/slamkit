"""
Unit tests for CosyVoiceFeatureExtractor with mocking.

These tests use mocking to test the feature extractor logic
without requiring the actual ONNX model file.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from slamkit.feature_extractor.cosyvoice_feature_extractor import CosyVoiceFeatureExtractor


class TestCosyVoiceFeatureExtractor:
    """Test suite for CosyVoiceFeatureExtractor"""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create a mock ONNX Runtime session"""
        mock_session = MagicMock()

        # Mock the inputs
        mock_input1 = MagicMock()
        mock_input1.name = 'mel_input'
        mock_input2 = MagicMock()
        mock_input2.name = 'length_input'

        mock_session.get_inputs.return_value = [mock_input1, mock_input2]

        # Mock the run method to return dummy tokens
        def mock_run(output_names, input_dict):
            # Get mel-spectrogram shape to determine output length
            mel_shape = input_dict['mel_input'].shape
            # Assuming mel has shape [batch, n_mels, time]
            time_dim = mel_shape[2] if len(mel_shape) > 2 else 100
            # Return dummy tokens (simulating 50Hz frame rate from mel)
            tokens = np.random.randint(0, 4096, size=(1, time_dim))
            return [tokens]

        mock_session.run = Mock(side_effect=mock_run)

        return mock_session

    @pytest.fixture
    def extractor(self, mock_onnx_session):
        """Create a CosyVoiceFeatureExtractor with mocked ONNX session"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session
                extractor = CosyVoiceFeatureExtractor(
                    onnx_path='dummy_model.onnx',
                    device='cpu',
                    num_units=4096,
                )
                return extractor

    def test_initialization(self, extractor):
        """Test feature extractor initialization"""
        assert extractor.onnx_path == 'dummy_model.onnx'
        assert extractor.device == 'cpu'
        assert extractor._sample_rate == 16000
        assert extractor._hop_length == 320

    def test_initialization_file_not_found(self):
        """Test that initialization fails when ONNX file doesn't exist"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="CosyVoice ONNX model not found"):
                CosyVoiceFeatureExtractor(onnx_path='nonexistent.onnx')

    def test_sample_rate_property(self, extractor):
        """Test sample_rate property"""
        assert extractor.sample_rate == 16000

    def test_get_unit_duration(self, extractor):
        """Test get_unit_duration method"""
        duration = extractor.get_unit_duration()
        expected_duration = 320 / 16000  # 0.02 seconds (50Hz)
        assert duration == pytest.approx(expected_duration)
        assert duration == pytest.approx(0.02)

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_extract_single_sample(self, mock_mel, extractor):
        """Test extracting features from a single audio sample"""
        # Create dummy audio (1 second at 16kHz)
        wav = torch.randn(1, 16000)

        # Mock mel-spectrogram output
        mock_mel_output = torch.randn(1, 128, 100)  # [batch, n_mels, time]
        mock_mel.return_value = mock_mel_output

        # Extract features
        features = extractor.extract(wav)

        # Assertions
        assert len(features) == 1  # One sample
        assert isinstance(features[0], np.ndarray)
        assert features[0].shape[0] > 0  # Has tokens
        mock_mel.assert_called_once()

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_extract_batch(self, mock_mel, extractor):
        """Test extracting features from a batch of audio samples"""
        # Create dummy audio batch (3 samples, 1 second each)
        batch_size = 3
        wav = torch.randn(batch_size, 16000)

        # Mock mel-spectrogram output
        mock_mel_output = torch.randn(1, 128, 100)
        mock_mel.return_value = mock_mel_output

        # Extract features
        features = extractor.extract(wav)

        # Assertions
        assert len(features) == batch_size
        assert all(isinstance(f, np.ndarray) for f in features)
        assert mock_mel.call_count == batch_size

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_extract_with_length_masking(self, mock_mel, extractor):
        """Test extracting features with length masking"""
        # Create dummy audio batch
        batch_size = 3
        wav = torch.randn(batch_size, 16000)
        lens = torch.tensor([16000, 8000, 4000])  # Different lengths

        # Mock mel-spectrogram output
        mock_mel_output = torch.randn(1, 128, 100)
        mock_mel.return_value = mock_mel_output

        # Extract features with length masking
        features = extractor.extract(wav, lens=lens)

        # Assertions
        assert len(features) == batch_size
        # Tokens should be truncated based on lengths
        # Expected lengths: 16000/320=50, 8000/320=25, 4000/320=13 (rounded up)
        assert features[0].shape[0] <= 51  # Allow some margin
        assert features[1].shape[0] <= 26
        assert features[2].shape[0] <= 14

    def test_extract_audio_too_long(self, extractor):
        """Test that extraction fails for audio longer than 30 seconds"""
        # Create 31 seconds of audio
        wav = torch.randn(1, 31 * 16000)

        with pytest.raises(ValueError, match="does not support extracting speech tokens for audio longer than 30s"):
            extractor.extract(wav)

    def test_extract_audio_exactly_30s(self, extractor):
        """Test that extraction works for exactly 30 seconds of audio"""
        # Create exactly 30 seconds of audio
        wav = torch.randn(1, 30 * 16000)

        # Mock mel-spectrogram
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram') as mock_mel:
            mock_mel_output = torch.randn(1, 128, 1500)
            mock_mel.return_value = mock_mel_output

            # Should not raise an error
            features = extractor.extract(wav)
            assert len(features) == 1

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_extract_empty_batch(self, mock_mel, extractor):
        """Test extracting features from an empty batch"""
        # Create empty batch
        wav = torch.randn(0, 16000)

        # Extract features
        features = extractor.extract(wav)

        # Should return empty list
        assert len(features) == 0

    def test_custom_sample_rate(self, mock_onnx_session):
        """Test initialization with custom sample rate"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                extractor = CosyVoiceFeatureExtractor(
                    onnx_path='dummy_model.onnx',
                    sample_rate=24000,
                    hop_length=480,
                )

                assert extractor.sample_rate == 24000
                assert extractor._hop_length == 480
                assert extractor.get_unit_duration() == pytest.approx(480 / 24000)

    def test_cuda_device_initialization(self, mock_onnx_session):
        """Test initialization with CUDA device"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.torch.cuda.is_available', return_value=True):
                    mock_inference.return_value = mock_onnx_session

                    extractor = CosyVoiceFeatureExtractor(
                        onnx_path='dummy_model.onnx',
                        device='cuda:0',
                    )

                    assert extractor.device == 'cuda:0'
                    # Verify CUDA provider was requested
                    call_args = mock_inference.call_args
                    providers = call_args[1]['providers']
                    assert "CUDAExecutionProvider" in providers

    def test_cpu_fallback_when_cuda_unavailable(self, mock_onnx_session):
        """Test that CPU provider is used when CUDA is not available"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.torch.cuda.is_available', return_value=False):
                    mock_inference.return_value = mock_onnx_session

                    extractor = CosyVoiceFeatureExtractor(
                        onnx_path='dummy_model.onnx',
                        device='cuda:0',  # Request CUDA but it's not available
                    )

                    # Verify CPU provider was used
                    call_args = mock_inference.call_args
                    providers = call_args[1]['providers']
                    assert "CPUExecutionProvider" in providers

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_token_range(self, mock_mel, extractor):
        """Test that extracted tokens are within expected range"""
        # Create dummy audio
        wav = torch.randn(1, 16000)

        # Mock mel-spectrogram output
        mock_mel_output = torch.randn(1, 128, 100)
        mock_mel.return_value = mock_mel_output

        # Extract features
        features = extractor.extract(wav)

        # Tokens should be in range [0, num_units)
        assert features[0].min() >= 0
        assert features[0].max() < 4096

    def test_kwargs_filtering(self, mock_onnx_session):
        """Test that None values in kwargs are filtered out"""
        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                # Pass some None kwargs (like from HuBERT configs)
                extractor = CosyVoiceFeatureExtractor(
                    onnx_path='dummy_model.onnx',
                    pretrained_model=None,
                    kmeans_path=None,
                    layer=None,
                )

                # Should initialize successfully
                assert extractor.onnx_path == 'dummy_model.onnx'


class TestCosyVoiceIntegration:
    """Integration tests that require the actual ONNX model"""

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
        reason="ONNX model not found (speech_tokenizer_v1.onnx or speech_tokenizer_v2.onnx)"
    )
    def test_real_extraction(self):
        """Test extraction with real ONNX model (if available)"""
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        # v1 has 4096 units, v2 has 8192 units
        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Create synthetic audio
        wav = torch.randn(1, 16000)  # 1 second

        # Extract features
        features = extractor.extract(wav)

        # Verify output
        assert len(features) == 1
        assert isinstance(features[0], np.ndarray)
        assert features[0].shape[0] > 0
        assert features[0].min() >= 0
        assert features[0].max() < expected_max_units

        print(f"✓ Real extraction test passed with {model_path}")
        print(f"  - Extracted {len(features[0])} tokens from 1 second of audio")
        print(f"  - Token range: [{features[0].min()}, {features[0].max()}]")
        print(f"  - Codebook size: {expected_max_units}")

    @pytest.mark.skipif(
        not __import__('os').path.exists('speech_tokenizer_v1.onnx') and
        not __import__('os').path.exists('speech_tokenizer_v2.onnx'),
        reason="ONNX model not found"
    )
    def test_real_extraction_with_actual_audio(self):
        """Test extraction with real audio file (if available)"""
        import torchaudio
        model_path = self._find_onnx_model()
        assert model_path is not None, "No ONNX model found"

        expected_max_units = 8192 if 'v2' in model_path else 4096

        extractor = CosyVoiceFeatureExtractor(
            onnx_path=model_path,
            device='cpu',
            num_units=expected_max_units,
        )

        # Try to load real audio
        try:
            wav, sr = torchaudio.load("example_data/audio/audio1.flac")
            # Convert to mono if stereo
            if wav.shape[0] > 1:
                wav = torch.mean(wav, dim=0, keepdim=True)
            # Resample if needed
            if sr != extractor.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, extractor.sample_rate)
            # Truncate to 5 seconds for testing
            max_len = 5 * extractor.sample_rate
            if wav.shape[1] > max_len:
                wav = wav[:, :max_len]
        except Exception:
            # Fall back to synthetic audio
            wav = torch.randn(1, 5 * 16000)  # 5 seconds

        # Extract features
        features = extractor.extract(wav)

        # Verify output
        assert len(features) == 1
        assert isinstance(features[0], np.ndarray)
        assert features[0].shape[0] > 0
        assert features[0].min() >= 0
        assert features[0].max() < expected_max_units

        # Calculate actual frame rate
        duration_sec = wav.shape[1] / extractor.sample_rate
        actual_frames = len(features[0])
        actual_frame_rate = actual_frames / duration_sec

        # Check that we got reasonable number of frames (between 20-60 Hz)
        assert actual_frames > 0
        assert 20 <= actual_frame_rate <= 60, f"Frame rate {actual_frame_rate:.1f}Hz is outside expected range 20-60Hz"

        print(f"✓ Real audio extraction test passed")
        print(f"  - Audio duration: {duration_sec:.2f}s")
        print(f"  - Extracted frames: {actual_frames}")
        print(f"  - Actual frame rate: {actual_frame_rate:.1f}Hz")
        print(f"  - Token range: [{features[0].min()}, {features[0].max()}]")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
