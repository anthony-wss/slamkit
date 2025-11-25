#!/bin/bash
# Train SFT model on ARC-Easy dataset
#
# Usage:
#   bash example_arc/train_arc.sh
#
# For quick testing (2 epochs, fewer steps):
#   bash example_arc/train_arc.sh --quick

set -e  # Exit on error

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in QUICK MODE (2 epochs for testing)"
fi

# Configuration
DATA_DIR="example_arc"
OUTPUT_DIR="${DATA_DIR}/arc_model"

# Training hyperparameters
if [ "$QUICK_MODE" = true ]; then
    NUM_EPOCHS=1
    SAVE_STRATEGY="no"
    EVAL_STRATEGY="no"
    LOGGING_STEPS=5
else
    NUM_EPOCHS=3
    SAVE_STRATEGY="epoch"
    EVAL_STRATEGY="epoch"
    LOGGING_STEPS=10
fi

echo "=========================================="
echo "Training SFT Model on ARC-Easy"
echo "=========================================="
echo "Train data: ${DATA_DIR}/arc_easy_train_tokens.jsonl"
echo "Val data: ${DATA_DIR}/arc_easy_validation_tokens.jsonl"
echo "Output dir: ${OUTPUT_DIR}"
echo "Epochs: ${NUM_EPOCHS}"
echo "=========================================="

# Run training
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/train_sft.py \
    data.train_path=${DATA_DIR}/arc_easy_train_tokens.jsonl \
    data.val_path=${DATA_DIR}/arc_easy_validation_tokens.jsonl \
    training_args.output_dir=${OUTPUT_DIR} \
    training_args.num_train_epochs=${NUM_EPOCHS} \
    training_args.per_device_train_batch_size=4 \
    training_args.gradient_accumulation_steps=4 \
    training_args.learning_rate=2e-5 \
    training_args.save_strategy=${SAVE_STRATEGY} \
    training_args.eval_strategy=${EVAL_STRATEGY} \
    training_args.logging_steps=${LOGGING_STEPS} \
    training_args.warmup_ratio=0.1 \
    training_args.dataloader_num_workers=4"

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Model saved to: ${OUTPUT_DIR}"
echo "=========================================="
