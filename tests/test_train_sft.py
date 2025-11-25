"""
Test script for train_sft.py CLI script.

This script tests:
1. Initializing SFT dataset from pre-tokenized data
2. Auto-determining vocab size from dataset
3. Initializing model with TWIST
4. Running training for a few steps
5. Verifying model checkpoints and outputs

Usage:
    # Run with Singularity container (recommended)
    singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
        bash -c "cd /workspace && python tests/test_train_sft.py"

    # Or run directly (requires slamkit installed)
    python tests/test_train_sft.py
"""

import json
import tempfile
import os
from pathlib import Path
import sys
import shutil

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import DictConfig
from hydra import initialize, compose
from slamkit.data import init_sft_dataset
from slamkit.model import tlm_factory
from slamkit.trainer import SLAMTrainer, SLAMTrainingArguments


def create_test_sft_data(num_samples=3):
    """Create test SFT token data (pre-tokenized)."""
    samples = []
    for i in range(num_samples):
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


def test_sft_dataset_init():
    """Test initializing SFT dataset."""
    print("\n" + "=" * 60)
    print("Test 1: Initialize SFT Dataset")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        samples = create_test_sft_data(num_samples=5)

        train_path = os.path.join(tmpdir, "train.jsonl")
        val_path = os.path.join(tmpdir, "val.jsonl")

        # Write train data
        with open(train_path, 'w') as f:
            for sample in samples[:3]:
                f.write(json.dumps(sample) + '\n')

        # Write val data
        with open(val_path, 'w') as f:
            for sample in samples[3:]:
                f.write(json.dumps(sample) + '\n')

        print(f"   ✓ Created test data ({len(samples)} samples)")

        # Initialize dataset
        cfg = DictConfig({
            'data': {
                'train_path': train_path,
                'val_path': val_path,
                'num_proc': 1
            }
        })

        dataset, collator = init_sft_dataset(cfg)

        print(f"   ✓ Dataset initialized")
        print(f"   - Train samples: {len(dataset['train'])}")
        print(f"   - Val samples: {len(dataset['validation'])}")

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

        print(f"   ✓ Dataset structure verified")
        print(f"   - Sample input_ids length: {len(sample['input_ids'])}")
        print(f"   - Sample labels length: {len(sample['labels'])}")
        print(f"   - User portion (first 20): all -100: {all(l == -100 for l in labels[:20])}")
        print(f"   - Assistant portion (last 30): contains non -100: {any(l != -100 for l in labels[20:])}")

        return tmpdir, train_path, val_path


def test_vocab_size_determination():
    """Test auto-determining vocab size from dataset."""
    print("\n" + "=" * 60)
    print("Test 2: Auto-determine Vocab Size")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
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

        train_path = os.path.join(tmpdir, "train.jsonl")
        with open(train_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

        # Load dataset
        cfg = DictConfig({
            'data': {
                'train_path': train_path,
                'val_path': train_path,
                'num_proc': 1
            }
        })

        dataset, _ = init_sft_dataset(cfg)

        # Determine vocab size (same logic as train_sft.py)
        determined_max_token = max(max(sample['input_ids']) for sample in dataset['train'].select(range(min(100, len(dataset['train'])))))
        determined_vocab_size = determined_max_token + 1

        print(f"   ✓ Max token ID found: {determined_max_token}")
        print(f"   ✓ Determined vocab size: {determined_vocab_size}")

        assert determined_max_token == max_token_id, f"Expected max token {max_token_id}, got {determined_max_token}"
        assert determined_vocab_size == max_token_id + 1, "Vocab size should be max_token_id + 1"

        print(f"   ✓ Vocab size determination is correct")


def test_model_initialization():
    """Test model initialization with TWIST."""
    print("\n" + "=" * 60)
    print("Test 3: Model Initialization with TWIST")
    print("=" * 60)

    # Create minimal model config
    model_cfg = DictConfig({
        'pretrained_model': None,
        'context_len': 512,
        'tlm_type': 'twist',
        'config_args': {
            'base_model_name': 'Qwen/Qwen2.5-0.5B',  # Use smaller model for testing
            'vocab_size': 1000,  # Small vocab for testing
            'use_cache': False,
            'torch_dtype': 'bfloat16',
            'trust_remote_code': True,
            'twist_init': True
        }
    })

    print(f"   Initializing model...")
    print(f"   - Base model: {model_cfg.config_args.base_model_name}")
    print(f"   - Vocab size: {model_cfg.config_args.vocab_size}")
    print(f"   - TWIST init: {model_cfg.config_args.twist_init}")

    model = tlm_factory(model_cfg)

    print(f"   ✓ Model initialized successfully")

    # Verify model structure
    assert hasattr(model, 'lm'), "Model should have 'lm' attribute"
    assert model.config.vocab_size == 1000, f"Expected vocab_size 1000, got {model.config.vocab_size}"

    # Verify embedding size
    embedding_size = model.get_input_embeddings().weight.shape[0]
    print(f"   ✓ Embedding layer size: {embedding_size}")
    assert embedding_size == 1000, f"Expected embedding size 1000, got {embedding_size}"

    print(f"   ✓ Model structure verified")


def test_training_pipeline():
    """Test end-to-end training pipeline."""
    print("\n" + "=" * 60)
    print("Test 4: End-to-End Training Pipeline")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        samples = create_test_sft_data(num_samples=4)

        train_path = os.path.join(tmpdir, "train.jsonl")
        val_path = os.path.join(tmpdir, "val.jsonl")

        with open(train_path, 'w') as f:
            for sample in samples[:3]:
                f.write(json.dumps(sample) + '\n')

        with open(val_path, 'w') as f:
            f.write(json.dumps(samples[3]) + '\n')

        # Initialize dataset
        cfg = DictConfig({
            'data': {
                'train_path': train_path,
                'val_path': val_path,
                'num_proc': 1
            }
        })

        dataset, collator = init_sft_dataset(cfg)
        print(f"   ✓ Dataset loaded ({len(dataset['train'])} train, {len(dataset['validation'])} val)")

        # Determine vocab size
        max_token = max(max(sample['input_ids']) for sample in dataset['train'])
        vocab_size = max_token + 1
        print(f"   ✓ Vocab size determined: {vocab_size}")

        # Initialize model
        model_cfg = DictConfig({
            'pretrained_model': None,
            'context_len': 512,
            'tlm_type': 'twist',
            'config_args': {
                'base_model_name': 'Qwen/Qwen2.5-0.5B',
                'vocab_size': vocab_size,
                'use_cache': False,
                'torch_dtype': 'bfloat16',
                'trust_remote_code': True,
                'twist_init': True
            }
        })

        model = tlm_factory(model_cfg)
        print(f"   ✓ Model initialized")

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

        print(f"   ✓ Training arguments created")
        print(f"   - Max steps: {train_args.max_steps}")
        print(f"   - Batch size: {train_args.per_device_train_batch_size}")

        # Create trainer
        trainer = SLAMTrainer(
            model=model,
            args=train_args,
            data_collator=collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
        )

        print(f"   ✓ Trainer created")

        # Run training
        print(f"   Running training for {train_args.max_steps} steps...")
        train_result = trainer.train()

        print(f"   ✓ Training completed successfully")
        print(f"   - Train loss: {train_result.training_loss:.4f}")
        print(f"   - Train steps: {train_result.global_step}")

        # Verify training ran
        assert train_result.global_step == 2, f"Expected 2 steps, got {train_result.global_step}"
        assert train_result.training_loss > 0, "Training loss should be positive"

        print(f"   ✓ Training verification passed")


def test_with_hydra_config():
    """Test using actual Hydra configuration."""
    print("\n" + "=" * 60)
    print("Test 5: Training with Hydra Config")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        samples = create_test_sft_data(num_samples=3)

        train_path = os.path.join(tmpdir, "train.jsonl")
        with open(train_path, 'w') as f:
            for sample in samples:
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
                    "model.config_args.base_model_name=Qwen/Qwen2.5-0.5B",  # Use smaller model
                ]
            )

        print(f"   ✓ Hydra config loaded")
        print(f"   - Train path: {cfg.data.train_path}")
        print(f"   - Model: {cfg.model.config_args.base_model_name}")
        print(f"   - Max steps: {cfg.training_args.get('max_steps', 'N/A')}")

        # Load dataset
        dataset, collator = init_sft_dataset(cfg)
        print(f"   ✓ Dataset loaded ({len(dataset['train'])} samples)")

        # Determine vocab size
        max_token = max(max(sample['input_ids']) for sample in dataset['train'].select(range(min(100, len(dataset['train'])))))
        cfg.model.config_args.vocab_size = max_token + 1
        print(f"   ✓ Vocab size set to {cfg.model.config_args.vocab_size}")

        # Initialize model
        model = tlm_factory(cfg.model)
        print(f"   ✓ Model initialized")

        # Create trainer
        from omegaconf import OmegaConf
        train_args = SLAMTrainingArguments(**OmegaConf.to_container(cfg.training_args))

        trainer = SLAMTrainer(
            model=model,
            args=train_args,
            data_collator=collator,
            train_dataset=dataset['train'],
            eval_dataset=dataset.get('validation', None),
        )

        print(f"   Running training...")
        train_result = trainer.train()

        print(f"   ✓ Training completed")
        print(f"   - Steps: {train_result.global_step}")
        print(f"   - Loss: {train_result.training_loss:.4f}")

        assert train_result.global_step == 2, "Should complete 2 steps"
        print(f"   ✓ Test passed")


def main():
    print("=" * 60)
    print("Testing train_sft.py")
    print("=" * 60)

    try:
        # Run tests
        test_sft_dataset_init()
        test_vocab_size_determination()
        test_model_initialization()
        test_training_pipeline()
        test_with_hydra_config()

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
