# SlamOmni SFT Data Conversion

This directory contains scripts and data for converting the SlamOmni dataset to SFT training format for finetuning Qwen3-0.6B.

## Overview

**Training Format:** `<question_text><assistant_text_response><assistant_cosyvoice_token>`

- **User turn:** Text-only questions (no audio)
- **Assistant turn:** Text response + CosyVoice speech tokens

## Data Statistics

- **Samples processed:** 500 (from train-00000-of-00853.parquet)
- **Average sequence length:** 834 tokens
- **Max sequence length:** 1,381 tokens
- **Min sequence length:** 334 tokens

## Vocabulary Configuration

- **Text vocabulary:** 151,669 tokens (Qwen/Qwen3-0.6B tokenizer)
- **Speech vocabulary:** 4,096 tokens (CosyVoice v1)
- **Total vocabulary:** 155,765 tokens
- **Speech token offset:** 151,669 (speech tokens start after text tokens)

## Data Format

Each sample in `sft_data/train.jsonl` contains:

```json
{
  "input_ids": [151644, 872, 198, ...],      // Full sequence (text + speech tokens)
  "labels": [-100, -100, ..., 9707, ...],    // -100 masks user turn, keeps assistant turn
  "attention_mask": [1, 1, 1, ...]           // All 1s
}
```

**ChatML Format:**
```
<|im_start|>user
{question_text}<|im_end|>
<|im_start|>assistant
{answer_text}
{cosyvoice_speech_tokens}<|im_end|>
```

**Label Masking:**
- User portion: Masked with -100 (no loss computed)
- Assistant text + speech: Unmasked (loss computed for both text and speech generation)

## Scripts

### 1. `inspect_data.py`
Inspects the original parquet file structure.

```bash
singularity exec --nv -B $PWD:/workspace -B /work/u3937558/data:/data \
  pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace/example_slamomni && python inspect_data.py"
```

### 2. `convert_to_sft.py`
Converts parquet data to SFT training format.

```bash
singularity exec --nv \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/data:/data \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace/example_slamomni && python convert_to_sft.py --num_samples 500"
```

**Arguments:**
- `--input_file`: Input parquet file (default: `/data/train-00000-of-00853.parquet`)
- `--output_file`: Output JSONL file (default: `./sft_data/train.jsonl`)
- `--num_samples`: Number of samples to process (default: 500)
- `--tokenizer`: HuggingFace tokenizer name (default: `Qwen/Qwen3-0.6B`)

### 3. `inspect_sample.py`
Inspects a converted sample to verify the format.

```bash
singularity exec --nv -B $PWD:/workspace \
  pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace/example_slamomni && python inspect_sample.py"
```

### 4. Training

Train the SFT model on the prepared data. The model uses TWIST initialization to adapt Qwen3-0.6B vocabulary for CosyVoice speech tokens.

**Debug training (1 epoch):**
```bash
singularity exec --nv -B $PWD:/workspace \
  pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/train_sft.py \
    data.train_path=example_slamomni/sft_data/train.jsonl \
    data.val_path=example_slamomni/sft_data/train.jsonl \
    training_args.output_dir=example_slamomni/checkpoints \
    training_args.num_train_epochs=1 \
    +training_args.save_steps=100 \
    training_args.logging_steps=10"
```

This command trains for 1 epoch on the 500-sample dataset (~125 steps with batch_size=4). Useful for:
- Verifying data pipeline and model initialization
- Checking GPU memory usage
- Testing loss convergence before full training


### 5. Inference

After running inference with the trained model, you can convert the generated speech tokens back to audio waveforms using the CosyVoice decoder.

#### 5-1. `test_inference.py`
Runs inference and generates both text and speech tokens.

```bash
singularity exec --nv \
    -B /work/u3937558/slamkit:/workspace \
    -B /work/u3937558/.cache:/tmp/cache \
    /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
    bash -c "export HF_HOME=/tmp/cache/huggingface && \
      export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
      export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
      cd /workspace/example_slamomni && python test_inference.py"
```

**Output:**
- `generated/test_response_tokens.json` - Generated speech tokens

#### 5-2. Token-to-WAV Conversion

Use the standalone `CosyVoice/token_to_wav.py` script to convert speech tokens to audio:

```bash
singularity exec --nv -B $PWD:/workspace \
  pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace/CosyVoice && source .venv/bin/activate && \
    python token_to_wav.py \
      --speech-tokens ../example_slamomni/generated/test_response_tokens.json \
      --prompt-audio ../example_data/audio/audio1.flac \
      --output ../example_slamomni/generated/test_response_audio.wav \
      --model-dir pretrained_models/CosyVoice-300M-SFT"
```

#### 5-3. Interact with the model

This is an terminal interface to interact with the model checkpoint.

```bash
singularity exec --nv -B $PWD:/workspace \
  pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && source .venv/bin/activate && \
    python example_slamomni/inference.py"
```

**Arguments:**
- `--tokens`: Path to JSON file with speech tokens (either `{"speech_tokens": [1234, 5678, ...]}`(dict) or `[1234, 5678]`(list))
- `--prompt-audio`: Reference audio for speaker characteristics (any sample rate, will be resampled to 16kHz)
- `--output`: Output WAV file path (24kHz mono audio)
- `--model-dir`: CosyVoice model directory (default: `pretrained_models/CosyVoice-300M-SFT`)
- `--speed`: Speed adjustment factor (optional, default: 1.0)

**How it works:**
1. **Prompt Processing**: Extracts speaker embedding and features from reference audio
2. **Flow Model**: Converts speech tokens → mel-spectrogram
3. **HiFT Vocoder**: Converts mel-spectrogram → audio waveform (24kHz)

**Example output:**
```
Loading tokens from: ../example_slamomni/generated/test_response_tokens.json
Loaded 438 tokens
Processing prompt audio: ../example_data/audio/audio1.flac
Extracting speaker embedding...
Extracting speech tokens...
Prompt processed: 704 tokens, 1320 mel frames
Converting 438 tokens to audio...
Running flow model (tokens → mel)...
Running HiFT vocoder (mel → audio)...
Generated 8.04s of audio
✅ Conversion completed successfully!
```

**Note:** The prompt audio provides the voice characteristics (speaker identity, pitch, timbre) for the generated speech. Use 3-10 seconds of clean speech from the desired speaker.

## Full training: text then speech

### 1. Process all data (~463k samples).
```bash
sbatch run_conversion.sh
```

#### 1.1 Split train and eval
* example_slamomni/sft_data/train_v1.jsonl (393611 samples)
* example_slamomni/sft_data/eval_v1.jsonl (69460 samples)

```bash
singularity exec --nv \
  -B /work/u3937558/slamkit:/workspace \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python scripts/split_sft_data.py \
    --input example_slamomni/sft_data/train_all.jsonl \
    --train_output example_slamomni/sft_data/train_v1.jsonl \
    --eval_output example_slamomni/sft_data/eval_v1.jsonl \
    --eval_ratio 0.15 \
    --seed 42"
```

### 2. Training
```bash
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
    data.train_path=example_slamomni/sft_data/train_v1.jsonl \
    data.val_path=example_slamomni/sft_data/eval_v1.jsonl \
    training_args.output_dir=example_slamomni/checkpoints \
    training_args.num_train_epochs=1 \
    training_args.learning_rate=1e-4 \
    training_args.logging_steps=10 \
    training_args.per_device_train_batch_size=16 \
    training_args.save_strategy=steps \
    logger.report_to=wandb \
    +logger.project=text_then_speech \
    +training_args.save_steps=1000 \
    training_args.eval_strategy=steps \
    training_args.eval_steps=1000"
```

To use wandb in singularity, run `wget https://curl.se/ca/cacert.pem` to get `cacert.pem`.

Use `group_by_length` will hang.

To save the new model right after initialization (for evaluation before training), uncomment these lines in `train_sft.py`:
```python
model.save_pretrained(train_args.output_dir)
exit()
``` 

#### 2.1 Training model v1.1

Modality token + freezing text parameters.

```bash
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
    data.train_path=example_slamomni/sft_data/train_v1.1.jsonl \
    data.val_path=example_slamomni/sft_data/eval_v1.1.jsonl \
    training_args.output_dir=example_slamomni/checkpoints/model_v1.1 \
    training_args.num_train_epochs=1 \
    training_args.learning_rate=1e-4 \
    training_args.logging_steps=10 \
    training_args.per_device_train_batch_size=16 \
    training_args.save_strategy=steps \
    logger.report_to=wandb \
    +logger.project=text_then_speech \
    +training_args.save_steps=1000 \
    training_args.eval_strategy=steps \
    training_args.eval_steps=1000 \
    freeze_text_embeddings=true"
```

### 3. Evaluation

#### 3.1 MMLU-Redux

```bash
singularity exec --nv \
  --env SSL_CERT_FILE=/workspace/cacert.pem \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace && python scripts/eval_mmlu_redux_zeroshot.py \
      --model_path example_slamomni/pretrain_qwen3-0.6B_checkpoint \
      --config high_school_mathematics \
      --output_dir outputs/mmlu_redux_eval \
      --bf16 \
      --quick"
```

#### 3.2 mmlu (via lm-evaluation-harness)

```bash
singularity exec --nv \
  --env SSL_CERT_FILE=/workspace/cacert.pem \
  -B /work/u3937558/slamkit:/workspace \
  -B /work/u3937558/.cache:/tmp/cache \
  /work/u3937558/slamkit/pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "export HF_HOME=/tmp/cache/huggingface && \
    export HF_DATASETS_CACHE=/tmp/cache/huggingface/datasets && \
    export TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers && \
    cd /workspace/lm-evaluation-harness && source venv/bin/activate && \
    lm_eval --model hf \
      --model_args pretrained=Qwen/Qwen3-0.6B \
      --tasks mmlu \
      --device cuda:0 \
      --batch_size 8"
```

#### 3.2.1 mmlu on fine-tuned checkpoint (via lm-evaluation-harness)

```bash
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
      --model_args pretrained=/workspace/example_slamomni/pretrain_qwen3-0.6B_checkpoint \
      --tasks mmlu_marketing \
      --device cuda:0 \
      --batch_size 8"
```

#### 3.2.2 generative mmlu-redux (via lm-evaluation-harness)

Add `--limit 0.01` for debuging.

```
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
      --tasks mmlu_redux_stem_generative \
      --device cuda:0 \
      --batch_size 8 \
      --output_path ./results/ \
      --log_samples"
```
