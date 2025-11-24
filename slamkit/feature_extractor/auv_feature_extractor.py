import os
import torch
import numpy as np
from typing import Optional

from .audio_feature_extractor import AudioFeatureExtractor

# Import AUV from the AUV submodule
import sys
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../AUV/src'))
from auv.model import AUV


class AUVFeatureExtractor(AudioFeatureExtractor):
    """
    AUV (Audio Universal Vector) Feature Extractor.

    Extracts 1-D codec tokens from audio using the AUV model.
    AUV uses a single quantizer (unlike multi-quantizer codecs) to represent audio.

    Args:
        checkpoint_path: Path to the AUV checkpoint file (e.g., 'auv.pt')
        compile: Whether to use torch.compile for faster inference (default: False)
        device: Device to run the model on (default: 'cuda:0')
        use_bf16: Whether to use bfloat16 for inference (default: False)
    """

    def __init__(
        self,
        checkpoint_path: str = 'auv.pt',
        compile: bool = False,
        device: str = 'cuda:0',
        use_bf16: bool = False,
        num_units: int = 20480,  # AUV codebook size (ignored, but required for config compatibility)
        **kwargs  # Ignore any extra parameters from config (e.g., pretrained_model, kmeans_path, layer from HuBERT configs)
    ):
        super().__init__()

        # Filter out None values from kwargs to avoid issues
        kwargs = {k: v for k, v in kwargs.items() if v is not None}

        self.checkpoint_path = checkpoint_path
        self.device = device
        self.use_bf16 = use_bf16 and torch.cuda.is_available()

        # Load the AUV model
        self.model = AUV()
        self.model.from_pretrained(checkpoint_path, device=device)
        self.model = self.model.to(device)  # Ensure all buffers are on the correct device
        self.model.eval()

        if compile:
            self.model = torch.compile(self.model, dynamic=True)

        # AUV configuration
        self._sample_rate = self.model.tokenizer.sample_rate  # 16000 Hz
        self._hop_length = self.model.tokenizer.hop_length    # 320 samples

    @torch.inference_mode()
    def extract(self, wav: torch.Tensor, lens: Optional[torch.Tensor] = None):
        """
        Extract 1-D codec tokens from audio.

        Args:
            wav: Audio tensor of shape [B, T] where B is batch size and T is time
            lens: Optional tensor of shape [B] containing the actual lengths of non-padded audio

        Returns:
            List of numpy arrays, each containing the codec tokens for one audio sample.
            Each array has shape [T'] where T' is the downsampled time dimension.
        """
        batch_size = wav.shape[0]
        results = []

        # Process each sample in the batch individually
        # (AUV encode only supports batch_size=1)
        for i in range(batch_size):
            sample = wav[i:i+1].to(self.device)

            # Prepare data dict for AUV
            data = {
                "sample": sample,
                "sample_rate": self._sample_rate,
            }

            # Extract tokens with optional autocast
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                enc_res = self.model.encode(data)
                tokens = enc_res["tokens"]  # Shape: [1, num_quantizers, T']

            # Convert to numpy and squeeze out batch and quantizer dimensions
            # tokens shape: [1, 1, T'] -> [T']
            tokens_np = tokens.squeeze(0).squeeze(0).cpu().numpy()

            # Handle length masking if provided
            if lens is not None:
                # Calculate the downsampled length
                original_len = lens[i].item()
                # AUV downsamples by hop_length
                downsampled_len = int(np.ceil(original_len / self._hop_length))
                tokens_np = tokens_np[:downsampled_len]

            results.append(tokens_np)

        return results

    def get_unit_duration(self) -> float:
        """
        Get the duration of each codec token in seconds.

        For AUV with 16kHz sample rate and hop_length of 320:
        Duration = 320 / 16000 = 0.02 seconds (50 Hz frame rate)

        Returns:
            Duration in seconds per token
        """
        return self._hop_length / self._sample_rate

    @property
    def sample_rate(self) -> int:
        """Sample rate expected by the AUV model (16kHz)"""
        return self._sample_rate
