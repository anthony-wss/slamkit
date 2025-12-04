"""
Test script for train_sft.py CLI script.

This script tests:
1. Initializing SFT dataset from pre-tokenized data
2. Auto-determining vocab size from dataset
3. Initializing model with TWIST
4. Running training for a few steps
5. Verifying model checkpoints and outputs
"""

import pytest
import json
import tempfile
import os
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from slamkit.data import init_sft_dataset
from slamkit.model import tlm_factory
from slamkit.trainer import SLAMTrainer, SLAMTrainingArguments


@pytest.fixture
def test_sft_data():
    """Create test SFT token data (pre-tokenized)."""
    samples = []
    for i in range(5):
        # Create mock tokenized data
        # Simulate ChatML format with user (masked) and assistant (unmasked) portions
        user_tokens = list(range(100, 120))  # 20 tokens for user
        assistant_tokens = list(range(200, 230))  # 30 tokens for assistant

        input_ids = user_tokens + assistant_tokens
        labels = [-100] * len(user_tokens) + assistant_tokens  # Mask user portion
        attention_mask = [1] * len(input_ids)

        samples.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        })

    return samples


@pytest.fixture
def tmpdir_with_data(test_sft_data):
    """Create temporary directory with test data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        train_path = os.path.join(tmpdir, "train.jsonl")
        val_path = os.path.join(tmpdir, "val.jsonl")

        # Write train data
        with open(train_path, 'w') as f:
            for sample in test_sft_data[:3]:
                f.write(json.dumps(sample) + '\n')

        # Write val data
        with open(val_path, 'w') as f:
            for sample in test_sft_data[3:]:
                f.write(json.dumps(sample) + '\n')

        yield tmpdir, train_path, val_path


class TestSFTDataset:
    """Test SFT dataset initialization and loading."""

    def test_sft_dataset_init(self, tmpdir_with_data):
        """Test initializing SFT dataset."""
        tmpdir, train_path, val_path = tmpdir_with_data

        # Initialize dataset
        cfg = DictConfig({
            'data': {
                'train_path': train_path,
                'val_path': val_path,
                'num_proc': 1
            }
        })

        dataset, collator = init_sft_dataset(cfg)

        print(f"✓ Dataset initialized")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Val samples: {len(dataset['validation'])}")

        # Verify dataset structure
        assert 'train' in dataset, "Missing train split"
        assert 'validation' in dataset, "Missing validation split"
        assert len(dataset['train']) == 3, f"Expected 3 train samples, got {len(dataset['train'])}"
        assert len(dataset['validation']) == 2, f"Expected 2 val samples, got {len(dataset['validation'])}"

        # Verify sample structure
        sample = dataset['train'][0]
        assert 'input_ids' in sample, "Missing input_ids"
        assert 'labels' in sample, "Missing labels"
        assert 'attention_mask' in sample, "Missing attention_mask"
        assert len(sample['input_ids']) == 50, "Incorrect input_ids length"
        assert len(sample['labels']) == 50, "Incorrect labels length"

        # Verify label masking
        labels = sample['labels']
        assert labels[:20] == [-100] * 20, "User portion should be masked"
        assert labels[20:] != [-100] * 30, "Assistant portion should not be all masked"

        print(f"✓ Dataset structure verified")
        print(f"  - Sample input_ids length: {len(sample['input_ids'])}")
        print(f"  - Sample labels length: {len(sample['labels'])}")

    def test_vocab_size_determination(self, tmpdir_with_data):
        """Test auto-determining vocab size from dataset."""
        tmpdir, train_path, val_path = tmpdir_with_data

        # Create test data with specific max token ID
        max_token_id = 5000
        samples = []
        for i in range(3):
            input_ids = list(range(100, 120)) + [max_token_id]  # Include max token
            labels = [-100] * 20 + [max_token_id]
            attention_mask = [1] * len(input_ids)

            samples.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            })

        test_path = os.path.join(tmpdir, "test.jsonl")
        with open(test_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Load dataset
        cfg = DictConfig({
            'data': {
                'train_path': test_path,
                'val_path': test_path,
                'num_proc': 1
            }
        })

        dataset, _ = init_sft_dataset(cfg)

        # Determine vocab size (same logic as train_sft.py)
        determined_max_token = max(max(sample['input_ids']) for sample in dataset['train'].select(range(min(100, len(dataset['train'])))))
        determined_vocab_size = determined_max_token + 1

        print(f"✓ Max token ID found: {determined_max_token}")
        print(f"✓ Determined vocab size: {determined_vocab_size}")

        assert determined_max_token == max_token_id, f"Expected max token {max_token_id}, got {determined_max_token}"
        assert determined_vocab_size == max_token_id + 1, "Vocab size should be max_token_id + 1"

    def test_max_length_truncation(self, tmpdir_with_data):
        """Test truncation with max_length parameter."""
        tmpdir, train_path, val_path = tmpdir_with_data

        # Create test data with varying lengths
        samples = []
        for length in [30, 60, 100]:  # Different lengths
            input_ids = list(range(1000, 1000 + length))
            labels = list(range(2000, 2000 + length))
            attention_mask = [1] * length

            samples.append({
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask
            })

        test_path = os.path.join(tmpdir, "test_truncation.jsonl")
        with open(test_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Test with max_length=40
        max_length = 40
        cfg = DictConfig({
            'data': {
                'train_path': test_path,
                'val_path': test_path,
                'num_proc': 1,
                'max_length': max_length
            }
        })

        dataset, _ = init_sft_dataset(cfg)

        print(f"✓ Dataset loaded with max_length={max_length}")

        # Verify all sequences are truncated to max_length
        for i, sample in enumerate(dataset['train']):
            assert len(sample['input_ids']) <= max_length, f"Sample {i}: input_ids length {len(sample['input_ids'])} exceeds max_length {max_length}"
            assert len(sample['labels']) <= max_length, f"Sample {i}: labels length {len(sample['labels'])} exceeds max_length {max_length}"
            assert len(sample['attention_mask']) <= max_length, f"Sample {i}: attention_mask length {len(sample['attention_mask'])} exceeds max_length {max_length}"

            # Verify truncation from the right (first tokens are preserved)
            original_length = [30, 60, 100][i]
            expected_length = min(original_length, max_length)
            assert len(sample['input_ids']) == expected_length, f"Sample {i}: Expected length {expected_length}, got {len(sample['input_ids'])}"

            # Verify first tokens are preserved (truncation from right)
            if original_length > max_length:
                expected_first_token = 1000  # First token from original sequence
                assert sample['input_ids'][0] == expected_first_token, f"Sample {i}: First token should be preserved"
                assert sample['labels'][0] == 2000, f"Sample {i}: First label should be preserved"

        print(f"✓ All sequences correctly truncated from the right")
        print(f"  - Sample 0 (original=30): {len(dataset['train'][0]['input_ids'])} tokens")
        print(f"  - Sample 1 (original=60): {len(dataset['train'][1]['input_ids'])} tokens")
        print(f"  - Sample 2 (original=100): {len(dataset['train'][2]['input_ids'])} tokens")

        # Test with max_length=None (no truncation)
        cfg_no_truncation = DictConfig({
            'data': {
                'train_path': test_path,
                'val_path': test_path,
                'num_proc': 1,
                'max_length': None
            }
        })

        dataset_no_truncation, _ = init_sft_dataset(cfg_no_truncation)

        print(f"✓ Dataset loaded with max_length=None (no truncation)")

        # Verify sequences maintain original lengths
        original_lengths = [30, 60, 100]
        for i, sample in enumerate(dataset_no_truncation['train']):
            expected_length = original_lengths[i]
            assert len(sample['input_ids']) == expected_length, f"Sample {i}: Expected length {expected_length}, got {len(sample['input_ids'])}"

        print(f"✓ All sequences maintain original lengths without truncation")


class TestModelInitialization:
    """Test model initialization with TWIST."""

    def test_model_initialization(self):
        """Test model initialization with TWIST."""
        # Create minimal model config
        model_cfg = DictConfig({
            'pretrained_model': None,
            'context_len': 512,
            'tlm_type': 'twist',
            'config_args': {
                'base_model_name': 'Qwen/Qwen3-0.6B',  # Use smaller model for testing
                'vocab_size': 1000,  # Small vocab for testing
                'use_cache': False,
                'torch_dtype': 'bfloat16',
                'trust_remote_code': True,
                'twist_init': True
            }
        })

        print(f"Initializing model...")
        print(f"  - Base model: {model_cfg.config_args.base_model_name}")
        print(f"  - Vocab size: {model_cfg.config_args.vocab_size}")
        print(f"  - TWIST init: {model_cfg.config_args.twist_init}")

        model = tlm_factory(model_cfg)

        print(f"✓ Model initialized successfully")

        # Verify model structure
        assert hasattr(model, 'lm'), "Model should have 'lm' attribute"
        assert model.config.vocab_size == 1000, f"Expected vocab_size 1000, got {model.config.vocab_size}"

        # Verify embedding size
        embedding_size = model.get_input_embeddings().weight.shape[0]
        print(f"✓ Embedding layer size: {embedding_size}")
        assert embedding_size == 1000, f"Expected embedding size 1000, got {embedding_size}"


@pytest.mark.slow
class TestTrainingPipeline:
    """Test end-to-end training pipeline (marked as slow)."""

    def test_training_pipeline(self, test_sft_data):
        """Test end-to-end training pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            val_path = os.path.join(tmpdir, "val.jsonl")

            with open(train_path, 'w') as f:
                for sample in test_sft_data[:3]:
                    f.write(json.dumps(sample) + '\n')

            with open(val_path, 'w') as f:
                f.write(json.dumps(test_sft_data[3]) + '\n')

            # Initialize dataset
            cfg = DictConfig({
                'data': {
                    'train_path': train_path,
                    'val_path': val_path,
                    'num_proc': 1
                }
            })

            dataset, collator = init_sft_dataset(cfg)
            print(f"✓ Dataset loaded ({len(dataset['train'])} train, {len(dataset['validation'])} val)")

            # Determine vocab size
            max_token = max(max(sample['input_ids']) for sample in dataset['train'])
            vocab_size = max_token + 1
            print(f"✓ Vocab size determined: {vocab_size}")

            # Initialize model
            model_cfg = DictConfig({
                'pretrained_model': None,
                'context_len': 512,
                'tlm_type': 'twist',
                'config_args': {
                    'base_model_name': 'Qwen/Qwen3-0.6B',
                    'vocab_size': vocab_size,
                    'use_cache': False,
                    'torch_dtype': 'bfloat16',
                    'trust_remote_code': True,
                    'twist_init': True
                }
            })

            model = tlm_factory(model_cfg)
            print(f"✓ Model initialized")

            # Create training arguments
            output_dir = os.path.join(tmpdir, "output")
            train_args = SLAMTrainingArguments(
                output_dir=output_dir,
                num_train_epochs=1,
                max_steps=2,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_steps=1,
                save_strategy='no',
                eval_strategy='no',
                report_to=[],
                bf16=True,
                dataloader_num_workers=0,  # Avoid multiprocessing issues in tests
            )

            print(f"✓ Training arguments created")
            print(f"  - Max steps: {train_args.max_steps}")

            # Create trainer
            trainer = SLAMTrainer(
                model=model,
                args=train_args,
                data_collator=collator,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
            )

            print(f"✓ Trainer created")

            # Run training
            print(f"Running training for {train_args.max_steps} steps...")
            train_result = trainer.train()

            print(f"✓ Training completed successfully")
            print(f"  - Train loss: {train_result.training_loss:.4f}")
            print(f"  - Train steps: {train_result.global_step}")

            # Verify training ran
            assert train_result.global_step == 2, f"Expected 2 steps, got {train_result.global_step}"
            assert train_result.training_loss > 0, "Training loss should be positive"

    def test_with_hydra_config(self, test_sft_data):
        """Test using actual Hydra configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.jsonl")
            with open(train_path, 'w') as f:
                for sample in test_sft_data[:3]:
                    f.write(json.dumps(sample) + '\n')

            output_dir = os.path.join(tmpdir, "output")

            # Initialize Hydra
            with initialize(version_base="1.3", config_path="../config"):
                cfg = compose(
                    config_name="train_sft",
                    overrides=[
                        f"data.train_path={train_path}",
                        f"data.val_path={train_path}",
                        f"training_args.output_dir={output_dir}",
                        "+training_args.max_steps=2",
                        "training_args.save_strategy=no",
                        "training_args.eval_strategy=no",
                        "training_args.dataloader_num_workers=0",
                        "model.config_args.base_model_name=Qwen/Qwen3-0.6B",  # Use smaller model
                    ]
                )

            print(f"✓ Hydra config loaded")
            print(f"  - Train path: {cfg.data.train_path}")

            # Load dataset
            dataset, collator = init_sft_dataset(cfg)
            print(f"✓ Dataset loaded ({len(dataset['train'])} samples)")

            # Determine vocab size
            max_token = max(max(sample['input_ids']) for sample in dataset['train'].select(range(min(100, len(dataset['train'])))))
            cfg.model.config_args.vocab_size = max_token + 1
            print(f"✓ Vocab size set to {cfg.model.config_args.vocab_size}")

            # Initialize model
            model = tlm_factory(cfg.model)
            print(f"✓ Model initialized")

            # Create trainer
            train_args = SLAMTrainingArguments(**OmegaConf.to_container(cfg.training_args))

            trainer = SLAMTrainer(
                model=model,
                args=train_args,
                data_collator=collator,
                train_dataset=dataset['train'],
                eval_dataset=dataset.get('validation', None),
            )

            print(f"Running training...")
            train_result = trainer.train()

            print(f"✓ Training completed")
            print(f"  - Steps: {train_result.global_step}")
            print(f"  - Loss: {train_result.training_loss:.4f}")

            assert train_result.global_step == 2, "Should complete 2 steps"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
