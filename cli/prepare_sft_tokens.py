from functools import partial
from multiprocessing.pool import ThreadPool
from omegaconf import DictConfig
import hydra
from tqdm import tqdm
from pathlib import Path
import json
import os
import logging
logger = logging.getLogger(__name__)

from slamkit.tokeniser import tokeniser_factory


def process_sft_sample(line, tokeniser, im_start_id, im_end_id, newline_id):
    """
    Process a single SFT sample and create training data with ChatML formatting.

    Input format:
    {
        "user_text": "...",
        "user_audio": {"units": [...], "duration": [...]},
        "assistant_text": "...",
        "assistant_audio": {"units": [...], "duration": [...]}
    }

    Output format (order: user text → user audio → assistant text → assistant audio):
    {
        "input_ids": [...],
        "labels": [...],  # -100 for user portion, actual tokens for assistant portion
        "audio_repr": "full text representation"
    }
    """
    try:
        cur = json.loads(line)

        # Stringify audio representations (use mode='test' to get pure unit tokens without interleaving)
        user_audio_str = tokeniser.stringify_representation([cur['user_audio']], mode='test')[0]
        assistant_audio_str = tokeniser.stringify_representation([cur['assistant_audio']], mode='test')[0]

        # Tokenize each component separately
        # <|im_start|>user\n{user_text}\n{user_audio}<|im_end|>\n<|im_start|>assistant\n{assistant_text}\n{assistant_audio}<|im_end|>

        user_text_tokens = tokeniser.text_tokeniser(cur['user_text'], add_special_tokens=False)['input_ids']
        user_audio_tokens = tokeniser.text_tokeniser(user_audio_str, add_special_tokens=False)['input_ids']
        assistant_text_tokens = tokeniser.text_tokeniser(cur['assistant_text'], add_special_tokens=False)['input_ids']
        assistant_audio_tokens = tokeniser.text_tokeniser(assistant_audio_str, add_special_tokens=False)['input_ids']

        # Build the sequence: <|im_start|>user\n{text}\n{audio}<|im_end|>\n<|im_start|>assistant\n{text}\n{audio}<|im_end|>
        input_ids = (
            [im_start_id] + tokeniser.text_tokeniser("user", add_special_tokens=False)['input_ids'] + [newline_id] +
            user_text_tokens + [newline_id] +
            user_audio_tokens + [im_end_id] + [newline_id] +
            [im_start_id] + tokeniser.text_tokeniser("assistant", add_special_tokens=False)['input_ids'] + [newline_id] +
            assistant_text_tokens + [newline_id] +
            assistant_audio_tokens + [im_end_id]
        )

        # Calculate where assistant portion starts (after the first <|im_end|>\n)
        user_portion_length = (
            1 +  # <|im_start|>
            len(tokeniser.text_tokeniser("user", add_special_tokens=False)['input_ids']) + 1 +  # "user" + \n
            len(user_text_tokens) + 1 +  # user_text + \n
            len(user_audio_tokens) + 1 + 1  # user_audio + <|im_end|> + \n
        )

        # Create labels: -100 for user portion, actual tokens for assistant
        labels = [-100] * user_portion_length + input_ids[user_portion_length:]

        # Add BOS/EOS tokens if tokenizer expects them
        if hasattr(tokeniser, 'bos_token_id') and tokeniser.bos_token_id is not None:
            input_ids = [tokeniser.bos_token_id] + input_ids + [tokeniser.eos_token_id]
            labels = [-100] + labels + [tokeniser.eos_token_id]

        # Create attention mask (all 1s)
        attention_mask = [1] * len(input_ids)

        # Prepare output
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

        return json.dumps(result)
    except Exception as e:
        import traceback
        logging.warning(f'Failed to process sample. Error: {e}')
        logging.debug(traceback.format_exc())
        return None


@hydra.main(config_name='prepare_sft_tokens', config_path='../config', version_base="1.3")
def prepare_sft_tokens(cfg: DictConfig):
    """
    Prepare SFT tokens from extracted features.

    This script:
    1. Loads SFT data with extracted audio features
    2. Formats data with ChatML structure (<|im_start|>user/assistant, <|im_end|>)
    3. Creates masked labels (only compute loss on assistant responses)
    4. Order: user text → user audio → assistant text → assistant audio
    """
    tokeniser = tokeniser_factory(cfg.tokeniser)

    # Add special tokens to tokenizer if not already present
    special_tokens = ["<|im_start|>", "<|im_end|>"]
    num_added = tokeniser.text_tokeniser.add_special_tokens(
        {'additional_special_tokens': special_tokens}
    )
    if num_added > 0:
        logger.info(f'Added {num_added} special tokens to tokenizer: {special_tokens}')

    # Get token IDs for special tokens
    im_start_id = tokeniser.text_tokeniser.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokeniser.text_tokeniser.convert_tokens_to_ids("<|im_end|>")

    # Get newline token ID (encode to handle different tokenizers)
    newline_tokens = tokeniser.text_tokeniser.encode("\n", add_special_tokens=False)
    newline_id = newline_tokens[0] if newline_tokens else tokeniser.text_tokeniser.convert_tokens_to_ids("<0x0A>")  # fallback for byte-level tokenizers

    logger.info(f'Special token IDs: <|im_start|>={im_start_id}, <|im_end|>={im_end_id}, \\n={newline_id}')

    # Create output directory if needed
    out_dir = os.path.dirname(cfg.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(cfg.out_path):
        logging.warning(f'{cfg.out_path} already exists. Deleting it!')
        os.remove(cfg.out_path)

    logging.info(f'Starting to prepare SFT tokens')
    with open(cfg.data_path, 'r') as f_in, open(cfg.out_path, 'a+') as f_out:
        with ThreadPool(cfg.n_threads) as p:
            process_fn = partial(process_sft_sample, tokeniser=tokeniser,
                                im_start_id=im_start_id, im_end_id=im_end_id, newline_id=newline_id)
            for jsonl in tqdm(p.imap(process_fn, f_in)):
                if jsonl:
                    f_out.write(jsonl + '\n')


if __name__ == '__main__':
    prepare_sft_tokens()
