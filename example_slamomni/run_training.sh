#!/bin/bash
#SBATCH --job-name=sft_train
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=48:00:00
#SBATCH --output=/work/u3937558/slamkit/example_slamomni/logs/training_%j.out
#SBATCH --error=/work/u3937558/slamkit/example_slamomni/logs/training_%j.err

# Create log directory
mkdir -p /work/u3937558/slamkit/example_slamomni/logs

# Create cache directory for HuggingFace datasets
mkdir -p /work/u3937558/.cache/huggingface

# Create checkpoint directory
mkdir -p /work/u3937558/slamkit/example_slamomni/checkpoints

# Load Singularity module
module load singularity/3.7.1

# Run SFT training
echo "Starting SFT training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

singularity exec --nv \
  --env SSL_CERT_FILE=/workspace/cacert.pem \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace && python cli/train_sft.py \
    model=text_then_speech \
    data.train_path=example_slamomni/sft_data/train_all.jsonl \
    data.val_path=example_slamomni/sft_data/train_500.jsonl \
    training_args.output_dir=example_slamomni/checkpoints \
    training_args.num_train_epochs=1 \
    training_args.learning_rate=1e-4 \
    training_args.logging_steps=10 \
    training_args.per_device_train_batch_size=16 \
    training_args.save_strategy=steps \
    logger.report_to=wandb \
    +logger.project=text_then_speech \
    +training_args.save_steps=1000"

echo "End time: $(date)"
echo "Training complete!"
