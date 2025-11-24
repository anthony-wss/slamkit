"""
Test script for extract_sft_features.py CLI script.

This script tests:
1. Loading SFT dataset with user and assistant audio
2. Extracting features from both user and assistant audio
3. Verifying the output format and field preservation
4. Testing batch processing

Usage:
    # Run with Singularity container (recommended)
    singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
        bash -c "cd /workspace && python tests/test_extract_sft_features.py"

    # Or run directly (requires slamkit installed)
    python tests/test_extract_sft_features.py
"""

import torch
import torchaudio
import json
import tempfile
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from slamkit.tokeniser import tokeniser_factory
from cli.extract_sft_features import SFTDataset, pad_collate_fn, extract_features


def create_test_audio(duration_sec=1.0, sample_rate=16000):
    """Create a simple sine wave audio for testing."""
    t = torch.linspace(0, duration_sec, int(duration_sec * sample_rate))
    # Create a sine wave at 440 Hz
    audio = torch.sin(2 * torch.pi * 440 * t)
    return audio.unsqueeze(0)  # Add channel dimension


def test_sft_dataset():
    """Test SFTDataset loading and preprocessing."""
    print("\n" + "=" * 60)
    print("Test 1: SFTDataset Loading")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio files
        user_audio = create_test_audio(duration_sec=1.0)
        assistant_audio = create_test_audio(duration_sec=1.5)

        user_audio_path = os.path.join(tmpdir, "user.wav")
        assistant_audio_path = os.path.join(tmpdir, "assistant.wav")

        torchaudio.save(user_audio_path, user_audio, 16000)
        torchaudio.save(assistant_audio_path, assistant_audio, 16000)

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

        # Test dataset loading
        dataset = SFTDataset(jsonl_path, sample_rate=16000)

        print(f"   ✓ Dataset created")
        print(f"   - Number of samples: {len(dataset)}")

        # Test __getitem__
        data, user_wav, user_len, asst_wav, asst_len = dataset[0]

        print(f"   ✓ Sample loaded successfully")
        print(f"   - User text: {data['user_text']}")
        print(f"   - Assistant text: {data['assistant_text']}")
        print(f"   - User audio shape: {user_wav.shape}")
        print(f"   - User audio length: {user_len}")
        print(f"   - Assistant audio shape: {asst_wav.shape}")
        print(f"   - Assistant audio length: {asst_len}")

        # Verify lengths
        assert user_len == user_wav.shape[0], "User audio length mismatch"
        assert asst_len == asst_wav.shape[0], "Assistant audio length mismatch"

        # Test subsample
        dataset.subsample_data(skip=0, take=1)
        assert len(dataset) == 1, "Subsample failed"

        print(f"   ✓ Subsample works correctly")

        return tmpdir, jsonl_path


def test_collate_fn():
    """Test the collate function for batching."""
    print("\n" + "=" * 60)
    print("Test 2: Batch Collation")
    print("=" * 60)

    # Create mock batch data
    batch = [
        ({"id": 1}, torch.randn(100), 100, torch.randn(150), 150),
        ({"id": 2}, torch.randn(80), 80, torch.randn(120), 120),
    ]

    data, wavs, lens = pad_collate_fn(batch)

    print(f"   ✓ Batch collated successfully")
    print(f"   - Batch size: {len(data)}")
    print(f"   - Concatenated wavs shape: {wavs.shape}")
    print(f"   - Expected: (2*batch_size, max_len) = ({2*len(batch)}, {max(150, 120)})")
    print(f"   - Lengths tensor: {lens}")

    # Verify shapes
    assert wavs.shape[0] == 2 * len(batch), "Batch size mismatch"
    assert len(lens) == 2 * len(batch), "Length tensor size mismatch"

    return True


def test_feature_extraction():
    """Test end-to-end feature extraction."""
    print("\n" + "=" * 60)
    print("Test 3: Feature Extraction Pipeline")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio files - use actual example files if available
        example_audio_path = "example_data/audio/audio1.flac"
        if os.path.exists(example_audio_path):
            print(f"   Using example audio: {example_audio_path}")
            user_audio_path = "example_data/audio/audio1.flac"
            assistant_audio_path = "example_data/audio/audio2.flac"
        else:
            print(f"   Creating synthetic audio")
            user_audio = create_test_audio(duration_sec=1.0)
            assistant_audio = create_test_audio(duration_sec=1.5)

            user_audio_path = os.path.join(tmpdir, "user.wav")
            assistant_audio_path = os.path.join(tmpdir, "assistant.wav")

            torchaudio.save(user_audio_path, user_audio, 16000)
            torchaudio.save(assistant_audio_path, assistant_audio, 16000)

        # Create test jsonl with multiple samples
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

        # Initialize tokeniser
        from omegaconf import DictConfig
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

        print(f"   Initializing tokeniser...")
        tokeniser = tokeniser_factory(tokeniser_cfg)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokeniser = tokeniser.to(device)
        print(f"   ✓ Tokeniser initialized on {device}")

        # Load dataset
        dataset = SFTDataset(jsonl_path, sample_rate=16000)
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, collate_fn=pad_collate_fn)

        print(f"   ✓ Dataset loaded: {len(dataset)} samples")

        # Extract features
        print(f"   Extracting features...")
        with open(output_path, 'w') as f:
            for data, wavs, lens in dataloader:
                wavs, lens = wavs.to(device), lens.to(device)
                batch_size = len(data)

                # Extract features
                tokenised = tokeniser.audio_represent(wavs, lens)

                # Split back into user and assistant
                user_tokenised = tokenised[:batch_size]
                assistant_tokenised = tokenised[batch_size:]

                # Write results
                for i, data_point in enumerate(data):
                    data_point['user_audio'] = user_tokenised[i]
                    data_point['assistant_audio'] = assistant_tokenised[i]
                    f.write(json.dumps(data_point) + '\n')

        print(f"   ✓ Features extracted to {output_path}")

        # Verify output
        print(f"\n   Verifying output format...")
        with open(output_path, 'r') as f:
            output_data = [json.loads(line) for line in f]

        print(f"   ✓ Output file loaded: {len(output_data)} samples")

        for i, sample in enumerate(output_data):
            print(f"\n   Sample {i}:")
            print(f"   - user_text: {sample['user_text']}")
            print(f"   - assistant_text: {sample['assistant_text']}")

            # Verify user_audio field
            assert 'user_audio' in sample, "Missing user_audio field"
            assert 'units' in sample['user_audio'], "Missing units in user_audio"
            assert 'duration' in sample['user_audio'], "Missing duration in user_audio"
            print(f"   - user_audio units: {len(sample['user_audio']['units'])} tokens")
            print(f"   - user_audio first 10 units: {sample['user_audio']['units'][:10]}")

            # Verify assistant_audio field
            assert 'assistant_audio' in sample, "Missing assistant_audio field"
            assert 'units' in sample['assistant_audio'], "Missing units in assistant_audio"
            assert 'duration' in sample['assistant_audio'], "Missing duration in assistant_audio"
            print(f"   - assistant_audio units: {len(sample['assistant_audio']['units'])} tokens")
            print(f"   - assistant_audio first 10 units: {sample['assistant_audio']['units'][:10]}")

            # Verify original fields are preserved
            assert 'user_audio_path' in sample, "user_audio_path not preserved"
            assert 'assistant_audio_path' in sample, "assistant_audio_path not preserved"

        print(f"\n   ✓ All output samples verified")

        return True


def main():
    print("=" * 60)
    print("Testing extract_sft_features.py")
    print("=" * 60)

    try:
        # Run tests
        test_sft_dataset()
        test_collate_fn()
        test_feature_extraction()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
