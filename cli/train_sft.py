import math
from omegaconf import OmegaConf, DictConfig
import hydra
import os

from slamkit.data import init_sft_dataset
from slamkit.model import tlm_factory
from slamkit.trainer import SLAMTrainer, SLAMTrainingArguments, RunTimeStopperCallback, MaxTokensStopperCallback
from slamkit.utils.init_utils import init_wandb, init_compile

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_name='train_sft', config_path='../config', version_base="1.3")
def main(cfg: DictConfig):
    """
    Fine-tune a language model on SFT (Supervised Fine-Tuning) data.

    This script expects pre-tokenized data with input_ids, labels, and attention_mask fields
    from prepare_sft_tokens.py. The model is fine-tuned on instruction-following dialogues
    with ChatML formatting.
    """
    # Load pre-tokenized dataset
    ds, collator = init_sft_dataset(cfg)
    logger.info('SFT dataset loaded')
    logger.info(f'Train samples: {len(ds["train"])}, Val samples: {len(ds.get("validation", []))}')

    if cfg.training_args.torch_compile:
        init_compile()
        logger.info('torch compile inited')

    # Initialize model
    if cfg.model.config_args.vocab_size == -1:
        # For SFT, we need to determine vocab size from the data
        # Get max token id from the dataset
        logger.info('Model vocab_size is -1, determining from dataset...')
        max_token_id = max(max(sample['input_ids']) for sample in ds['train'].select(range(min(100, len(ds['train'])))))
        cfg.model.config_args.vocab_size = max_token_id + 1
        logger.info(f'Set model vocab_size to {cfg.model.config_args.vocab_size}')

    model = tlm_factory(cfg.model)
    logger.info('model inited')

    # Note: Model vocab_size is already set correctly during initialization
    # No need to resize embeddings here as TWIST handles it during model creation

    if cfg.training_args.get('warmup_steps', 0) > 0 and cfg.training_args.get('warmup_ratio', .0) > 0:
        logger.warning('Both warmup_steps and warmup_ratio are set, setting to maximum of the two!')
        # this calculation is somewhat approximate, but should be good enough
        bs = cfg.training_args.per_device_train_batch_size * cfg.training_args.gradient_accumulation_steps * int(os.environ.get("WORLD_SIZE", 1))
        n_steps = math.ceil(len(ds['train']) / bs) * cfg.training_args.num_train_epochs
        if n_steps * cfg.training_args.warmup_ratio > cfg.training_args.warmup_steps:
            cfg.training_args.warmup_steps = 0

    train_args = SLAMTrainingArguments(**OmegaConf.to_container(cfg.training_args))

    if cfg.logger.report_to == 'wandb':
        name = os.path.basename(os.path.normpath(cfg.training_args.output_dir))

        if int(os.environ.get('RANK', 0)) == 0:
            init_wandb(cfg, name)

        train_args.run_name = name
        train_args.report_to = ['wandb']
        logger.info('wandb inited')
    else:
        train_args.report_to = []

    callbacks = None
    if cfg.get("run_time", None) is not None:
        callbacks = [RunTimeStopperCallback(cfg.run_time)]
    if cfg.get("train_max_tokens", None) is not None:
        callback = MaxTokensStopperCallback(cfg.train_max_tokens)
        if callbacks is None:
            callbacks = [callback]
        else:
            callbacks.append(callback)

    trainer = SLAMTrainer(
        model=model,
        args=train_args,
        data_collator=collator,
        train_dataset=ds['train'],
        eval_dataset=ds.get('validation', None),
        callbacks=callbacks
    )

    trainer.train(resume_from_checkpoint=cfg.cont_training)


if __name__ == '__main__':
    main()
