# ARC-Easy Text-Only SFT Training Example

This directory contains a complete example of training and evaluating a text-only SFT model on the ARC-Easy dataset using SlamKit.

**Note**: This example uses **pure text** (no audio) to demonstrate SFT training on a standard NLP benchmark. It serves as a reference for how to adapt SlamKit for multimodal or text-only tasks.

## Dataset

**ARC-Easy** (AI2 Reasoning Challenge - Easy): Multiple choice science questions from Allen AI
- Train: 2,251 examples
- Validation: 570 examples
- Test: 2,376 examples
- Format: 4-choice questions (A, B, C, D)
- Random baseline: 25%

## Pipeline Overview

```
1. Download & Format    2. Tokenize          3. Train             4. Evaluate
   ARC-Easy        →    Text-Only SFT   →    SFT Model       →    Test Accuracy
   (prepare_arc_data.py) (prepare_arc_tokens.py) (train_sft.py)      (eval_arc.py)
```

## Step-by-Step Instructions

### Step 1: Prepare Dataset

Download ARC-Easy from HuggingFace and convert to text-only SFT format:

```bash
python example_arc/prepare_arc_data.py
```

**Output files**:
- `example_arc/arc_easy_train.jsonl`: 2,251 training conversations
- `example_arc/arc_easy_validation.jsonl`: 570 validation conversations
- `example_arc/arc_easy_test.jsonl`: 2,376 test conversations

**Output format** (per line):
```json
{
  "user_text": "Multiple Choice question: Which factor will most likely cause a person to develop a fever?\n- a leg muscle relaxing after exercise=A\n- a bacterial population in the bloodstream=B\n- several viral particles on the skin=C\n- carbohydrates being digested in the stomach=D\n\nRespond only with the letter of the correct answer.",
  "assistant_text": "B"
}
```

### Step 2: Tokenize Data

Tokenize conversations with ChatML formatting and label masking:

```bash
# Tokenize training set
python example_arc/prepare_arc_tokens.py \
    --data_path example_arc/arc_easy_train.jsonl \
    --out_path example_arc/arc_easy_train_tokens.jsonl \
    --tokenizer Qwen/Qwen3-0.6B

# Tokenize validation set
python example_arc/prepare_arc_tokens.py \
    --data_path example_arc/arc_easy_validation.jsonl \
    --out_path example_arc/arc_easy_validation_tokens.jsonl \
    --tokenizer Qwen/Qwen3-0.6B
```

**Output format** (per line):
```json
{
  "input_ids": [151644, 882, 198, ...],
  "labels": [-100, -100, ..., 33],
  "attention_mask": [1, 1, 1, ...]
}
```

**Key features**:
- ChatML format: `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n{text}<|im_end|>`
- Label masking: User portion masked with -100 (no loss), only train on assistant response
- Average tokens per sample: ~77 tokens

### Step 3: Train Model

Train SFT model using SlamKit's `train_sft.py`:

```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/train_sft.py \
    data.train_path=example_arc/arc_easy_train_tokens.jsonl \
    data.val_path=example_arc/arc_easy_validation_tokens.jsonl \
    training_args.output_dir=example_arc/arc_model \
    training_args.num_train_epochs=3 \
    training_args.per_device_train_batch_size=4 \
    training_args.gradient_accumulation_steps=4 \
    training_args.learning_rate=2e-5 \
    training_args.save_strategy=epoch \
    training_args.eval_strategy=epoch \
    training_args.logging_steps=10"
```

**Training configuration**:
- Base model: Qwen/Qwen3-0.6B with TWIST initialization
- Learning rate: 2e-5 with cosine schedule
- Batch size: 4 per device × 4 gradient accumulation = 16 effective batch size
- Epochs: 3
- Precision: bfloat16
- Training time: ~3 minutes on one H100

**Model checkpoints** saved to `example_arc/arc_model/`

### Step 4: Evaluate Model

#### Evaluate Base Model (No Training)

First, establish a baseline by evaluating the pretrained Qwen3-0.6B without any SFT training:

```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python example_arc/eval_arc_base.py \
    --model_name Qwen/Qwen3-0.6B \
    --data_path example_arc/arc_easy_test.jsonl \
    --batch_size 16 \
    --max_problems 100"
```

#### Evaluate SFT-Trained Model

Evaluate the fine-tuned model on ARC-Easy test set:

```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python example_arc/eval_arc.py \
    --model_path example_arc/arc_model/checkpoint-423 \
    --data_path example_arc/arc_easy_test.jsonl \
    --batch_size 16"
```

**Quick evaluation** (first 100 problems):
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python example_arc/eval_arc.py \
    --model_path example_arc/arc_model/checkpoint-423 \
    --data_path example_arc/arc_easy_test.jsonl \
    --batch_size 16 \
    --max_problems 100"
```

**Evaluation method**:
- Categorical evaluation (efficient batch processing)
- Extract logits at answer position
- Narrow to valid letter tokens (A, B, C, D)
- Take argmax to get prediction
- Compare with ground truth

**Actual results** (100 test problems):

| Model | Accuracy | Improvement |
|-------|----------|-------------|
| Random baseline | 25.00% | - |
| Base Qwen3-0.6B (no training) | **27.66%** | +2.66% over random |
| After 3 epochs SFT | **79.79%** | **+52.13%** over base |

The SFT training achieves a massive **52 percentage point improvement** over the base model, demonstrating the effectiveness of supervised fine-tuning for teaching the model to follow the multiple-choice format and answer science questions correctly.

## File Structure

```
example_arc/
├── README.md                           # This file
├── prepare_arc_data.py                 # Step 1: Download and format dataset
├── prepare_arc_tokens.py               # Step 2: Tokenize for training
├── train_arc.sh                        # Step 3: Training script
├── eval_arc_base.py                    # Step 4a: Evaluate base model (no training)
├── eval_arc.py                         # Step 4b: Evaluate trained model
├── arc_easy_train.jsonl                # Training conversations (2,251)
├── arc_easy_validation.jsonl           # Validation conversations (570)
├── arc_easy_test.jsonl                 # Test conversations (2,376)
├── arc_easy_train_tokens.jsonl         # Tokenized training data
├── arc_easy_validation_tokens.jsonl    # Tokenized validation data
└── arc_model/                          # Trained model checkpoints (after training)
```

## Notes

### Text-Only vs Multimodal

This example is **text-only** (no audio). For multimodal SFT training with both speech and text:
1. Use `cli/extract_sft_features.py` to extract audio features
2. Use `cli/prepare_sft_tokens.py` to combine text and audio tokens
3. Use `cli/train_sft.py` with multimodal data

See `CLAUDE.md` for full multimodal SFT pipeline documentation.

### Customization

**Change base model**:
```bash
# Edit config/model/sft_qwen.yaml
base_model_name: Qwen/Qwen2.5-0.5B  # or any other compatible model
```

**Training hyperparameters**:
- Adjust learning rate: `training_args.learning_rate=1e-5`
- Adjust epochs: `training_args.num_train_epochs=5`
- Adjust batch size: `training_args.per_device_train_batch_size=8`

**Data mixing**:
To mix ARC-Easy with other datasets, combine tokenized jsonl files:
```bash
cat example_arc/arc_easy_train_tokens.jsonl \
    other_dataset_tokens.jsonl \
    > mixed_train_tokens.jsonl
```

## Reference

This example is inspired by [nanochat](https://github.com/karpathy/nanochat)'s approach to training on ARC-Easy, adapted for SlamKit's architecture.

**nanochat d20 results** (~$100 model, 4e19 FLOPs):
- ARC-Easy: 35.61%
- After SFT: 38.76%

## Troubleshooting

**Out of memory**:
- Reduce batch size: `training_args.per_device_train_batch_size=2`
- Increase gradient accumulation: `training_args.gradient_accumulation_steps=8`

**Slow tokenization**:
- Tokenization is single-threaded but fast (~3000 samples/sec)
- Pre-tokenize once, reuse for multiple training runs

**Evaluation issues**:
- Ensure special tokens `<|im_start|>` and `<|im_end|>` are in tokenizer
- Check that letter tokens (A, B, C, D) are single tokens
- Verify model checkpoint path is correct
