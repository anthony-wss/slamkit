#!/bin/bash
#SBATCH --job-name=mmlu_eval
#SBATCH --partition=normal
#SBATCH --account=MST111038
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/work/u3937558/slamkit/example_slamomni/logs/conversion_%j.out
#SBATCH --error=/work/u3937558/slamkit/example_slamomni/logs/conversion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=anthonywang903@gmail.com

# Load Singularity module
module load singularity/3.7.1

# Run conversion for ALL data
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
    cd /workspace/lm-evaluation-harness && source venv/bin/activate && \
    lm_eval --model unitlm \
      --model_args pretrained=/workspace/example_slamomni/checkpoints/text_then_speech_lr_1e-4/checkpoint-6151 \
      --tasks mmlu \
      --device cuda:0 \
      --batch_size 8"

echo "End time: $(date)"
echo "Complete!"
