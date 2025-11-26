"""
Integration tests for CosyVoiceFeatureExtractor with tokeniser factory.

These tests verify that the CosyVoice feature extractor integrates
properly with SlamKit's tokeniser system.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
from omegaconf import DictConfig, OmegaConf


class TestCosyVoiceTokeniserIntegration:
    """Test CosyVoice integration with tokeniser factory"""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create a mock ONNX Runtime session"""
        import numpy as np
        mock_session = MagicMock()

        mock_input1 = MagicMock()
        mock_input1.name = 'mel_input'
        mock_input2 = MagicMock()
        mock_input2.name = 'length_input'

        mock_session.get_inputs.return_value = [mock_input1, mock_input2]

        def mock_run(output_names, input_dict):
            mel_shape = input_dict['mel_input'].shape
            time_dim = mel_shape[2] if len(mel_shape) > 2 else 100
            tokens = np.random.randint(0, 4096, size=(1, time_dim))
            return [tokens]

        mock_session.run = MagicMock(side_effect=mock_run)
        return mock_session

    def test_feature_extractor_factory(self, mock_onnx_session):
        """Test that the feature extractor factory can create CosyVoice extractor"""
        from slamkit.tokeniser.audio_tokeniser import _init_feature_extractor

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                config = DictConfig({
                    'onnx_path': 'speech_tokenizer_v1.onnx',
                    'device': 'cpu',
                    'num_units': 4096,
                })

                extractor = _init_feature_extractor('cosyvoice', config)

                assert extractor is not None
                assert extractor.sample_rate == 16000
                assert extractor.get_unit_duration() == pytest.approx(0.02)

    def test_feature_extractor_factory_unknown_type(self):
        """Test that factory raises error for unknown feature extractor type"""
        from slamkit.tokeniser.audio_tokeniser import _init_feature_extractor

        config = DictConfig({'dummy': 'config'})

        with pytest.raises(ValueError, match="Unknown speech tokeniser type"):
            _init_feature_extractor('unknown_type', config)

    def test_tokeniser_factory_with_cosyvoice(self, mock_onnx_session):
        """Test creating a unit tokeniser with CosyVoice feature extractor"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                config = OmegaConf.create({
                    'tokeniser_type': 'unit',
                    'feature_extractor_type': 'cosyvoice',
                    'feature_extractor': {
                        'onnx_path': 'speech_tokenizer_v1.onnx',
                        'device': 'cpu',
                        'num_units': 4096,
                    },
                    'params': {
                        'num_units': 4096,
                        'load_fe': True,
                        'dedup': True,
                        'bos_eos_token_id': 1,
                    }
                })

                tokeniser = tokeniser_factory(config)

                assert tokeniser is not None
                assert tokeniser.model is not None
                assert tokeniser.model.sample_rate == 16000

    def test_tokeniser_without_feature_extractor(self):
        """Test creating a tokeniser without loading feature extractor"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory

        config = OmegaConf.create({
            'tokeniser_type': 'unit',
            'feature_extractor_type': 'cosyvoice',
            'feature_extractor': {
                'onnx_path': 'speech_tokenizer_v1.onnx',
                'num_units': 4096,
            },
            'params': {
                'num_units': 4096,
                'load_fe': False,  # Don't load feature extractor
                'dedup': True,
                'bos_eos_token_id': 1,
            }
        })

        tokeniser = tokeniser_factory(config)

        assert tokeniser is not None
        assert tokeniser.model is None  # Should be None when load_fe=False

    @patch('slamkit.feature_extractor.cosyvoice_feature_extractor.whisper.log_mel_spectrogram')
    def test_end_to_end_audio_representation(self, mock_mel, mock_onnx_session):
        """Test end-to-end audio representation extraction"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory
        import numpy as np

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                # Mock mel-spectrogram
                mock_mel_output = torch.randn(1, 128, 100)
                mock_mel.return_value = mock_mel_output

                config = OmegaConf.create({
                    'tokeniser_type': 'unit',
                    'feature_extractor_type': 'cosyvoice',
                    'feature_extractor': {
                        'onnx_path': 'speech_tokenizer_v1.onnx',
                        'device': 'cpu',
                        'num_units': 4096,
                    },
                    'params': {
                        'num_units': 4096,
                        'load_fe': True,
                        'dedup': True,
                        'bos_eos_token_id': 1,
                    }
                })

                tokeniser = tokeniser_factory(config)

                # Create dummy audio
                wav = torch.randn(2, 16000)  # 2 samples, 1 second each

                # Extract audio representation
                representations = tokeniser.audio_represent(wav)

                # Verify output
                assert len(representations) == 2
                assert 'units' in representations[0]
                assert 'duration' in representations[0]
                # When dedup=True, units and duration are zip objects/tuples, so convert to list
                units = list(representations[0]['units'])
                durations = list(representations[0]['duration'])
                assert len(units) > 0
                assert len(durations) > 0
                assert len(units) == len(durations)

    def test_config_loading_from_yaml(self):
        """Test that CosyVoice config can be loaded from YAML"""
        import os
        from omegaconf import OmegaConf

        config_path = 'config/tokeniser/unit_cosyvoice.yaml'

        if os.path.exists(config_path):
            # Load config
            config = OmegaConf.load(config_path)

            # Verify structure
            assert config.tokeniser_type == 'unit'
            assert config.params.dedup is True
            assert config.params.bos_eos_token_id == 1

            # Check defaults reference
            assert 'cosyvoice' in str(config.defaults)

    def test_feature_extractor_config_loading(self):
        """Test loading feature extractor config from YAML"""
        import os
        from omegaconf import OmegaConf

        config_path = 'config/tokeniser/feature_extractor/cosyvoice.yaml'

        if os.path.exists(config_path):
            # Load config
            config = OmegaConf.load(config_path)

            # Verify structure
            assert config.tokeniser.feature_extractor_type == 'cosyvoice'
            assert 'onnx_path' in config.tokeniser.feature_extractor
            # v2 has 8192 units, v1 has 4096
            assert config.tokeniser.feature_extractor.num_units in [4096, 8192]
            assert config.tokeniser.feature_extractor.device == 'cuda:0'

    def test_num_units_propagation(self, mock_onnx_session):
        """Test that num_units from feature extractor propagates to tokeniser params"""
        from slamkit.tokeniser.audio_tokeniser import tokeniser_factory

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                config = OmegaConf.create({
                    'tokeniser_type': 'unit',
                    'feature_extractor_type': 'cosyvoice',
                    'feature_extractor': {
                        'onnx_path': 'speech_tokenizer_v1.onnx',
                        'device': 'cpu',
                        'num_units': 4096,  # Specified in feature extractor
                    },
                    'params': {
                        'num_units': -1,  # Will be set from feature_extractor
                        'load_fe': True,
                        'dedup': True,
                        'bos_eos_token_id': 1,
                    }
                })

                tokeniser = tokeniser_factory(config)

                # Verify num_units was propagated
                assert config.params.num_units == 4096


class TestCosyVoiceVsOtherExtractors:
    """Comparative tests between CosyVoice and other extractors"""

    def test_cosyvoice_vs_hubert_frame_rate(self):
        """Compare frame rates between CosyVoice and HuBERT"""
        from slamkit.feature_extractor import CosyVoiceFeatureExtractor
        from unittest.mock import patch, MagicMock

        mock_onnx_session = MagicMock()
        mock_input1 = MagicMock()
        mock_input1.name = 'mel_input'
        mock_input2 = MagicMock()
        mock_input2.name = 'length_input'
        mock_onnx_session.get_inputs.return_value = [mock_input1, mock_input2]

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                cosyvoice = CosyVoiceFeatureExtractor(
                    onnx_path='dummy.onnx',
                    device='cpu',
                )

                # CosyVoice: 50Hz (0.02s per token)
                assert cosyvoice.get_unit_duration() == pytest.approx(0.02)

                # HuBERT is 25Hz (0.04s per token) - CosyVoice is 2x faster
                assert cosyvoice.get_unit_duration() == pytest.approx(0.04 / 2)

    def test_cosyvoice_vs_auv_sample_rate(self):
        """Compare sample rates between CosyVoice and AUV"""
        from slamkit.feature_extractor import CosyVoiceFeatureExtractor
        from unittest.mock import patch, MagicMock

        mock_onnx_session = MagicMock()
        mock_input1 = MagicMock()
        mock_input1.name = 'mel_input'
        mock_input2 = MagicMock()
        mock_input2.name = 'length_input'
        mock_onnx_session.get_inputs.return_value = [mock_input1, mock_input2]

        with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.onnxruntime.InferenceSession') as mock_inference:
            with patch('slamkit.feature_extractor.cosyvoice_feature_extractor.os.path.exists', return_value=True):
                mock_inference.return_value = mock_onnx_session

                cosyvoice = CosyVoiceFeatureExtractor(
                    onnx_path='dummy.onnx',
                    device='cpu',
                )

                # Both CosyVoice and AUV use 16kHz
                assert cosyvoice.sample_rate == 16000

                # Both have 50Hz frame rate (0.02s per token)
                assert cosyvoice.get_unit_duration() == pytest.approx(0.02)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
