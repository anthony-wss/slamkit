# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SlamKit is an open-source toolkit for training and evaluating Speech Language Models (SLMs). It supports:
- Speech-only pre-training
- Preference alignment (DPO)
- Speech-text interleaving
- Multiple evaluation metrics and generation capabilities

The codebase is built on HuggingFace Transformers and uses Hydra for configuration management.

## Development Commands

By default, we run all commands inside a Singularity container on HPC/Slurm environments. The container provides Python development headers needed for `torch.compile` and CUDA support.

### Initial Setup

```bash
# Load Singularity module
module load singularity/3.7.1

# Pull PyTorch container with CUDA support (one-time, 6.3GB download)
singularity pull docker://pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Install dependencies in container (one-time setup)
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && pip install -e . && pip install soundfile"
```

**Note**: We use `pip` instead of `uv` inside the container for convenience. Packages install to `~/.local/` which persists across sessions due to automatic home directory bind mounting.

**Optional dependencies**: Some evaluation metrics require additional packages:
- `nltk`: Required for `asr_perplexity` metric. Install with: `pip install nltk`

### Running Commands

All Python commands should be run inside the container:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/SCRIPT.py [args]"
```

For convenience, you can create an alias:
```bash
alias srun='singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif bash -c "cd /workspace && "'
# Then use: srun python cli/extract_features.py [args]"
```

### Core Pipeline (4 main stages)

1. **Extract features** - Convert audio to discrete tokens using speech tokenizers:

**HuBERT-based (500 units, 25Hz):**
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/extract_features.py data_path=<AUDIO_DIR> ext=<flac|wav> out_path=<OUTPUT>.jsonl batch_size=16 tokeniser=unit_hubert_25 tokeniser.feature_extractor.compile=true num_workers=4"
```

**AUV-based (20480 units, 50Hz):**
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/extract_features.py data_path=<AUDIO_DIR> ext=<flac|wav> out_path=<OUTPUT>.jsonl batch_size=4 tokeniser=unit_auv tokeniser.feature_extractor.checkpoint_path=auv.pt num_workers=0"
```
**Note**: AUV requires the `auv.pt` checkpoint file. Download from [AUV repository](https://github.com/ishine/AUV) and place in the project root. AUV uses a larger codebook (20480 vs 500) and higher frame rate (50Hz vs 25Hz) than HuBERT.

2. **Prepare tokens** - Create string representations for training:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/prepare_tokens.py data_path=<FEATURES>.jsonl out_path=<OUT_DIR>"
```
**Note**: `out_path` is a directory. The script creates the output file inside it with the same name as the input file. For example, if `data_path=example_data/features.jsonl` and `out_path=example_data/tokens`, the output will be `example_data/tokens/features.jsonl`.

3. **Train** - Pre-train a speech language model:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/train.py data.train_path=<OUT_DIR>/<FEATURES>.jsonl data.val_path=<OUT_DIR>/<FEATURES>.jsonl tokeniser=unit_hubert_25 training_args.num_train_epochs=1 training_args.output_dir=<OUT_DIR>"
```
**Note**: `data.train_path` should point to the output file from step 2. Following the example above, if step 2 used `out_path=example_data/tokens`, then use `data.train_path=example_data/tokens/features.jsonl`.

4. **Eval** - Evaluate or generate continuations:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/eval.py tokeniser=unit_hubert_25 metric=generate batch_size=2 model.pretrained_model=<MODEL_PATH> metric.data_path=<AUDIO>/*.flac metric.num_files=2 metric.used_token_modality=SPEECH vocoder=vocoder_hubert_25 metric.out_path=<OUTPUT_DIR> metric.generate_kwargs.do_sample=false metric.generate_kwargs.max_new_tokens=50"
```
**Note**:
- The vocoder requires downloading checkpoints to `~/.textless/`. See the Vocoders section for details.
- `metric.used_token_modality=SPEECH` must be set when using a vocoder to generate audio.
- For greedy decoding (deterministic), use `metric.generate_kwargs.do_sample=false`.
- Generated audio files are saved to `metric.out_path` directory.

### Preference Alignment

Feature extraction for preference alignment (expects jsonl with prompt_path, chosen_path, rejected_path):
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/preference_alignment_feature_extractor.py data_path=<DATA>.jsonl out_path=<OUTPUT>.jsonl"
```

Train with DPO:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/preference_alignment_train.py data.train_path=<DATA>.jsonl model.pretrained_model=<MODEL_PATH>"
```

### Supervised Fine-Tuning (SFT)

**Note**: SFT support is newly added to SlamKit to enable speech-text joint training on instruction-following dialogues.

SFT enables training on instruction-following dialogues with both speech and text modalities. The pipeline processes conversation pairs (user → assistant) into training-ready tokens with proper label masking.

**Input data format** (JSONL with conversation pairs):
```json
{"user_text": "Hello, how are you", "user_audio_path": "path/to/user.flac", "assistant_text": "I'm fine, thank you.", "assistant_audio_path": "path/to/assistant.flac"}
```

**Pipeline structure**:
```
audio/ + sft_test.jsonl → sft_features.jsonl → sft_tokens.jsonl
```

1. **Extract SFT features** - Process both user and assistant audio:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/extract_sft_features.py data_path=<SFT_DATA>.jsonl out_path=<OUTPUT>.jsonl batch_size=8 tokeniser=unit_hubert_25"
```

**Input**: JSONL file with `user_text`, `user_audio_path`, `assistant_text`, `assistant_audio_path` fields
**Output**: JSONL file with added `user_audio` and `assistant_audio` feature fields (units and durations)

2. **Prepare SFT tokens** - Create ChatML-formatted training sequences:
```bash
singularity exec --nv -B $PWD:/workspace pytorch_2.6.0-cuda12.4-cudnn9-devel.sif \
  bash -c "cd /workspace && python cli/prepare_sft_tokens.py data_path=<SFT_FEATURES>.jsonl out_path=<OUTPUT>.jsonl"
```

**Input**: JSONL file from step 1 with extracted features
**Output**: Training-ready JSONL with `input_ids`, `labels`, and `attention_mask`

**Key features**:
- **ChatML formatting**: Sequences formatted as `<|im_start|>user\n{text}\n{audio}<|im_end|>\n<|im_start|>assistant\n{text}\n{audio}<|im_end|>`
- **Token order**: user text → user audio → assistant text → assistant audio
- **Label masking**: User portion masked with -100 (no loss), only assistant responses contribute to training loss
- **Tokenizer**: Uses `interleaved_hubert_25` tokenizer for speech-text handling

**Notes**:
- The script automatically adds special tokens (`<|im_start|>`, `<|im_end|>`) to the tokenizer vocabulary
- Handles different text tokenizers (e.g., Qwen, OPT) with proper newline encoding
- Output file is ready for standard HuggingFace training pipeline

## Architecture Overview

### Configuration System (Hydra-based)
All scripts use Hydra configurations located in `config/`. Key config groups:
- **tokeniser**: `unit_hubert_25`, `unit_auv`, `interleaved_hubert_25` (speech-only vs speech-text)
- **model**: `slam`, `twist`, `gslm` (different model architectures)
- **training_args**: Standard HuggingFace TrainingArguments
- **metric**: Evaluation metrics (generate, tstorycloze, sblimp, swuggy, salmon, etc.)
- **vocoder**: For converting tokens back to audio (vocoder_hubert_25)
- **logger**: wandb or print

### Core Components

**Tokenizers** (`slamkit/tokeniser/`):
- `AudioTokeniser`: Abstract base class for all tokenizers
- `UnitTokeniser`: Speech-only tokenization (discrete units)
- `InterleavingTokeniser`: Speech-text interleaving with configurable methods (poisson, uniform)
- Pipeline: `audio_represent()` → `stringify_representation()` → `string_tokenise()`

**Models** (`slamkit/model/`):
- `UnitLM`: Core speech language model (wraps HuggingFace CausalLM)
- `SpeechLM`: Handles generation and evaluation
- `TokenLM`: Factory for creating models from configs
- All models built on top of pre-trained text LMs (e.g., OPT, Qwen, Llama)
- Supports TWIST initialization (vocabulary adaptation for speech tokens)

**Feature Extractors** (`slamkit/feature_extractor/`):
- `AudioFeatureExtractor`: Base class
- `HubertFeatureExtractor`: Extracts features from audio using HuBERT/mHuBERT (500 units, 25Hz)
  - Converts raw audio to discrete tokens via k-means clustering on continuous features
- `AUVFeatureExtractor`: Extracts features using AUV neural codec (20480 units, 50Hz)
  - Uses learned vector quantization instead of k-means clustering
  - Higher codebook size and frame rate than HuBERT

**Trainers** (`slamkit/trainer/`):
- `SLAMTrainer`: Custom trainer extending HuggingFace Trainer
- `SLAMDPOTrainer`: DPO-specific trainer
- Callbacks: `RunTimeStopperCallback`, `MaxTokensStopperCallback`

**Data** (`slamkit/data/`):
- `init_dataset()`: Creates HuggingFace datasets from jsonl files
- Supports packing (requires flash_attention_2)
- Glob patterns supported for multiple files: `data.train_path=path/**/*.jsonl`

**Metrics** (`slamkit/metric/`):
- Modelling metrics: sBLIMP, sWUGGY, sStoryCloze, tStoryCloze, SALMon
- Generative metrics: GenPPL, Auto-BLEU
- Cross-modal generation and evaluation

**Vocoders** (`slamkit/vocoder/`):
- Converts discrete tokens back to audio
- Isolated code from textlesslib (HiFi-GAN based)
- Set `TEXTLESS_CHECKPOINT_ROOT` to customize download location (defaults to `~/.textless/`)

### Key Design Patterns

1. **Feature Sharing**: Extract features once, reuse across different tokenizers with same feature extractor
2. **Hydra Overrides**: Use `+training_args.param=value` for HuggingFace TrainingArguments
3. **Multi-file datasets**: Use glob patterns for train/val paths to merge shards
4. **Tokenizer-Model Matching**: For interleaved models, text_tokeniser_path must match model base_model_name
5. **Packing**: Only supported with flash_attention_2 (`model.config_args.attn_implementation=flash_attention_2`)

## Important Notes

### Training
- Default scheduler is cosine - requires good estimate of total steps. Use `+training_args.max_steps=N` when using early stopping via `run_time=24:00`
- For packing: Must use `data.packing=true` with `model.config_args.attn_implementation=flash_attention_2`
- Warmup: If both warmup_steps and warmup_ratio set, maximum is used
- Model vocab_size=-1 auto-sets to tokenizer vocab size

### Feature Extraction
- Processes entire audio files without splitting - use VAD for long files (>30 min) to avoid OOM
- Files sorted by length (decreasing) to minimize padding and fail early on OOM
- Use `data_skip=N data_take=M` to subset data
- Use `tokeniser.feature_extractor.compile=true` for torch.compile (faster but higher init latency)
- **AUV-specific notes**:
  - Requires `auv.pt` checkpoint file (download from AUV repository)
  - Set `num_workers=0` due to AUV model initialization constraints
  - Use smaller batch sizes (e.g., `batch_size=4`) compared to HuBERT
  - Configure checkpoint path: `tokeniser.feature_extractor.checkpoint_path=auv.pt`
  - Optional: Enable bfloat16 with `tokeniser.feature_extractor.use_bf16=true`

### Evaluation
- Generation config matches paper defaults
- For non-DPO models, may want `metric.generate_kwargs.repetition_penalty=1.0`
- Only one metric per eval.py run (use Hydra multirun for multiple)
- Generated files saved to `metric.out_path` directory (defaults to `generated/`)
- **Vocoder setup**: Vocoders download checkpoints to `~/.textless/` automatically. If download fails due to SSL issues, manually download the required files:
  - For `vocoder_hubert_25` (mHuBERT-based):
    - https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/hifigan_lj_mhubert_base_25hz.pt
    - https://dl.fbaipublicfiles.com/textless_nlp/twist/speech_tokenizer/hifigan_lj_mhubert_base_25hz_config.json

### Library Usage
The toolkit can be used as a library:
```python
from slamkit.model import UnitLM
model = UnitLM.from_pretrained("slprl/slam_scaled")
model.push_to_hub('<entity>/model_name')  # HuggingFace integration
```

## Related Work
- **SLAM**: Speech-only pre-training on 1 GPU - see `docs/SLAM.md`
- **SIMS**: Scaling analysis of interleaved speech-text models - see `docs/SIMS.md`
