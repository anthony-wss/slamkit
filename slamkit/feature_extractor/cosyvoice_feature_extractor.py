import os
import torch
import numpy as np
from typing import Optional
import onnxruntime
import whisper

from .audio_feature_extractor import AudioFeatureExtractor


class CosyVoiceFeatureExtractor(AudioFeatureExtractor):
    """
    CosyVoice Speech Tokenizer Feature Extractor.

    Extracts discrete speech tokens from audio using CosyVoice's ONNX-based speech tokenizer.
    The tokenizer converts audio to mel-spectrogram and then to discrete tokens.

    Args:
        onnx_path: Path to the CosyVoice speech tokenizer ONNX model (e.g., 'speech_tokenizer_v1.onnx' or 'speech_tokenizer_v2.onnx')
        device: Device to run the model on (default: 'cuda:0')
        num_units: Number of units in the codebook (informational, default: 4096)
        sample_rate: Expected audio sample rate (default: 16000)
        hop_length: Hop length for frame rate calculation (default: 320(v1), gives 50Hz frame rate)
                    For v2 model, hop_length is 640, gives 25Hz frame rate.
    """

    def __init__(
        self,
        onnx_path: str = 'speech_tokenizer_v1.onnx',
        device: str = 'cuda:0',
        num_units: int = 4096,  # CosyVoice codebook size (informational)
        sample_rate: int = 16000,
        **kwargs  # Ignore any extra parameters from config
    ):
        super().__init__()

        # Filter out None values from kwargs to avoid issues
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self.onnx_path = onnx_path
        self.device = device
        self._sample_rate = sample_rate
        if "v1" in onnx_path:
            self._hop_length = 320
        elif "v2" in onnx_path:
            self._hop_length = 640
        else:
            raise ValueError(f"Unsupported CosyVoice model version in ONNX path {onnx_path}. Use v1 or v2 model.")

        # Verify that the ONNX model file exists
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"CosyVoice ONNX model not found at {onnx_path}. "
                f"Please download the model from the CosyVoice repository."
            )

        # Initialize ONNX Runtime session
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1

        # Configure GPU execution if available
        self.use_cuda = torch.cuda.is_available() and 'cuda' in device
        if self.use_cuda:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.ort_session = onnxruntime.InferenceSession(
            onnx_path,
            sess_options=option,
            providers=providers
        )

        # Verify which execution provider is being used
        actual_providers = self.ort_session.get_providers()
        if self.use_cuda and "CUDAExecutionProvider" not in actual_providers:
            print(f"Warning: CUDA requested but not available. Falling back to: {actual_providers}")

    @torch.inference_mode()
    def extract(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None):
        """
        Extract discrete speech tokens from audio.

        Args:
            wav: Audio tensor of shape [B, T] where B is batch size and T is time
            lens: Optional tensor of shape [B] containing the actual lengths of non-padded audio

        Returns:
            List of numpy arrays, each containing the speech tokens for one audio sample.
            Each array has shape [T'] where T' is the downsampled time dimension.
        """
        batch_size = wav.shape[0]
        results = []

        # Process each sample in the batch individually
        for i in range(batch_size):
            sample = wav[i:i+1]  # Keep as [1, T]

            # Check audio length (CosyVoice typically limits to 30s)
            audio_duration = sample.shape[1] / self._sample_rate
            if audio_duration > 30:
                raise ValueError(
                    f"CosyVoice does not support extracting speech tokens for audio longer than 30s. "
                    f"Current audio duration: {audio_duration:.2f}s"
                )

            # Extract mel-spectrogram using Whisper's method (128 mel bins)
            # This matches the CosyVoice frontend implementation
            feat = whisper.log_mel_spectrogram(sample, n_mels=128)

            # Run ONNX model to extract speech tokens
            # Input: mel-spectrogram and its length
            # Output: discrete speech tokens
            # Note: ONNX Runtime with CUDA provider requires CPU inputs but runs on GPU internally
            speech_token = self.ort_session.run(
                None,
                {
                    self.ort_session.get_inputs()[0].name: feat.detach().cpu().numpy(),
                    self.ort_session.get_inputs()[1].name: np.array([feat.shape[2]], dtype=np.int32)
                }
            )[0].flatten()

            # Handle length masking if provided
            if lens is not None:
                # Calculate the downsampled length
                original_len = lens[i].item()
                # CosyVoice downsamples by hop_length
                downsampled_len = int(np.ceil(original_len / self._hop_length))
                speech_token = speech_token[:downsampled_len]

            results.append(speech_token)

        return results

    def get_unit_duration(self) -> float:
        """
        Get the duration of each speech token in seconds.

        For CosyVoice v1 with 16kHz sample rate and hop_length of 320:
        Duration = 320 / 16000 = 0.02 seconds (50 Hz frame rate)

        Returns:
            Duration in seconds per token
        """
        return self._hop_length / self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the CosyVoice model (16kHz)"""
        return self._sample_rate
