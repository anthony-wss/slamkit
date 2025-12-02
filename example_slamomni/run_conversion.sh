#!/bin/bash
#SBATCH --job-name=convert_sft
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/u3937558/slamkit/example_slamomni/logs/conversion_%j.out
#SBATCH --error=/work/u3937558/slamkit/example_slamomni/logs/conversion_%j.err

# Create log directory
mkdir -p /work/u3937558/slamkit/example_slamomni/logs

# Create cache directory for HuggingFace datasets
mkdir -p /work/u3937558/.cache/huggingface

# Load Singularity module
module load singularity/3.7.1

# Run conversion for ALL data
echo "Starting conversion of all parquet files..."
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

singularity exec --nv \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/data:/data \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace/example_slamomni && python convert_to_sft.py \
    --input_file '/data/*.parquet' \
    --output_file ./sft_data/train_all.jsonl"

echo "End time: $(date)"
echo "Conversion complete!"
