"""
Test script for extract_sft_features.py CLI script.

This script tests:
1. Loading SFT dataset with user and assistant audio
2. Extracting features from both user and assistant audio
3. Verifying the output format and field preservation
4. Testing batch processing
"""

import pytest
import torch
import torchaudio
import json
import tempfile
import os
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from slamkit.tokeniser import tokeniser_factory
from cli.extract_sft_features import SFTDataset, pad_collate_fn


def create_test_audio(duration_sec=1.0, sample_rate=16000):
    """Create a simple sine wave audio for testing."""
    t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate))
    # Create a sine wave at 440 Hz
    audio = torch.sin(2 * torch.pi * 440 * t)
    return audio.unsqueeze(0)  # Add channel dimension


@pytest.fixture
def test_audio_files():
    """Create temporary test audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        user_audio = create_test_audio(duration_sec=1.0)
        assistant_audio = create_test_audio(duration_sec=1.5)

        user_audio_path = os.path.join(tmpdir, "user.wav")
        assistant_audio_path = os.path.join(tmpdir, "assistant.wav")

        torchaudio.save(user_audio_path, user_audio, 16000)
        torchaudio.save(assistant_audio_path, assistant_audio, 16000)

        yield tmpdir, user_audio_path, assistant_audio_path


@pytest.fixture
def test_jsonl_file(test_audio_files):
    """Create test JSONL file."""
    tmpdir, user_audio_path, assistant_audio_path = test_audio_files

    # Create test jsonl
    test_data = {
        "user_text": "Hello, how are you?",
        "user_audio_path": user_audio_path,
        "assistant_text": "I'm fine, thank you.",
        "assistant_audio_path": assistant_audio_path
    }

    jsonl_path = os.path.join(tmpdir, "test.jsonl")
    with open(jsonl_path, 'w') as f:
        f.write(json.dumps(test_data) + '\n')

    return tmpdir, jsonl_path


@pytest.fixture
def hubert_tokeniser():
    """Create HuBERT tokeniser for testing."""
    tokeniser_cfg = DictConfig({
        'tokeniser_type': 'unit',
        'feature_extractor_type': 'hubert',
        'params': {
            'dedup': True,
            'pad_token_id': 0,
            'num_units': None,
            'load_fe': True,
        },
        'feature_extractor': {
            'pretrained_model': 'facebook/hubert-base-ls960',
            'layer': 9,
            'kmeans_path': 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960_L9_km500.bin',
            'num_units': 500,
            'cache_path': None,
            'compile': False,
            'load_config_only': False,
        }
    })

    tokeniser = tokeniser_factory(tokeniser_cfg)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokeniser = tokeniser.to(device)
    return tokeniser


class TestSFTDataset:
    """Test SFTDataset loading and preprocessing."""

    def test_sft_dataset_loading(self, test_jsonl_file):
        """Test SFTDataset loading and preprocessing."""
        tmpdir, jsonl_path = test_jsonl_file

        # Test dataset loading
        dataset = SFTDataset(jsonl_path, sample_rate=16000)

        print(f"✓ Dataset created")
        print(f"  - Number of samples: {len(dataset)}")

        assert len(dataset) == 1, "Should have 1 sample"

        # Test __getitem__
        data, user_wav, user_len, asst_wav, asst_len = dataset[0]

        print(f"✓ Sample loaded successfully")
        print(f"  - User text: {data['user_text']}")
        print(f"  - Assistant text: {data['assistant_text']}")
        print(f"  - User audio shape: {user_wav.shape}")
        print(f"  - Assistant audio shape: {asst_wav.shape}")

        # Verify lengths
        assert user_len == user_wav.shape[0], "User audio length mismatch"
        assert asst_len == asst_wav.shape[0], "Assistant audio length mismatch"
        assert 'user_text' in data
        assert 'assistant_text' in data

    def test_subsample(self, test_audio_files):
        """Test subsample functionality."""
        tmpdir, user_audio_path, assistant_audio_path = test_audio_files

        # Create jsonl with 3 samples
        jsonl_path = os.path.join(tmpdir, "test_multi.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(3):
                test_data = {
                    "user_text": f"Question {i}",
                    "user_audio_path": user_audio_path,
                    "assistant_text": f"Answer {i}",
                    "assistant_audio_path": assistant_audio_path
                }
                f.write(json.dumps(test_data) + '\n')

        dataset = SFTDataset(jsonl_path, sample_rate=16000)
        assert len(dataset) == 3

        # Test subsample
        dataset.subsample_data(skip=0, take=1)
        assert len(dataset) == 1, "Subsample failed"

        print(f"✓ Subsample works correctly")


class TestCollateFn:
    """Test the collate function for batching."""

    def test_collate_fn(self):
        """Test the collate function for batching."""
        # Create mock batch data
        batch = [
            ({"id": 1}, torch.randn(100), 100, torch.randn(150), 150),
            ({"id": 2}, torch.randn(80), 80, torch.randn(120), 120),
        ]

        data, wavs, lens = pad_collate_fn(batch)

        print(f"✓ Batch collated successfully")
        print(f"  - Batch size: {len(data)}")
        print(f"  - Concatenated wavs shape: {wavs.shape}")
        print(f"  - Lengths tensor: {lens}")

        # Verify shapes
        assert wavs.shape[0] == 2 * len(batch), "Batch size mismatch"
        assert len(lens) == 2 * len(batch), "Length tensor size mismatch"


@pytest.mark.slow
class TestFeatureExtraction:
    """Test end-to-end feature extraction (marked as slow)."""

    def test_feature_extraction_pipeline(self, test_audio_files, hubert_tokeniser):
        """Test end-to-end feature extraction."""
        tmpdir, user_audio_path, assistant_audio_path = test_audio_files

        # Create jsonl with multiple samples
        jsonl_path = os.path.join(tmpdir, "test_input.jsonl")
        with open(jsonl_path, 'w') as f:
            for i in range(3):
                test_data = {
                    "user_text": f"User message {i}",
                    "user_audio_path": user_audio_path,
                    "assistant_text": f"Assistant response {i}",
                    "assistant_audio_path": assistant_audio_path
                }
                f.write(json.dumps(test_data) + '\n')

        output_path = os.path.join(tmpdir, "test_output.jsonl")

        print(f"✓ Tokeniser initialized")

        # Load dataset
        dataset = SFTDataset(jsonl_path, sample_rate=16000)
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate_fn)

        print(f"✓ Dataset loaded: {len(dataset)} samples")

        # Extract features
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Extracting features on {device}...")

        with open(output_path, 'w') as f:
            for data, wavs, lens in dataloader:
                wavs, lens = wavs.to(device), lens.to(device)
                batch_size = len(data)

                # Extract features
                tokenised = hubert_tokeniser.audio_represent(wavs, lens)

                # Split back into user and assistant
                user_tokenised = tokenised[:batch_size]
                assistant_tokenised = tokenised[batch_size:]

                # Write results
                for i, data_point in enumerate(data):
                    data_point['user_audio'] = user_tokenised[i]
                    data_point['assistant_audio'] = assistant_tokenised[i]
                    f.write(json.dumps(data_point) + '\n')

        print(f"✓ Features extracted to {output_path}")

        # Verify output
        print(f"\nVerifying output format...")
        with open(output_path, 'r') as f:
            output_data = [json.loads(line) for line in f]

        print(f"✓ Output file loaded: {len(output_data)} samples")

        for i, sample in enumerate(output_data):
            # Verify user_audio field
            assert 'user_audio' in sample, "Missing user_audio field"
            assert 'units' in sample['user_audio'], "Missing units in user_audio"
            assert 'duration' in sample['user_audio'], "Missing duration in user_audio"

            # Verify assistant_audio field
            assert 'assistant_audio' in sample, "Missing assistant_audio field"
            assert 'units' in sample['assistant_audio'], "Missing units in assistant_audio"
            assert 'duration' in sample['assistant_audio'], "Missing duration in assistant_audio"

            # Verify original fields are preserved
            assert 'user_audio_path' in sample, "user_audio_path not preserved"
            assert 'assistant_audio_path' in sample, "assistant_audio_path not preserved"

            print(f"  Sample {i}: ✓")

        print(f"\n✓ All output samples verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
