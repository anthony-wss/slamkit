import logging
from typing import Optional
logger = logging.getLogger(__name__)

import json
import torchaudio
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import hydra
from omegaconf import DictConfig

from slamkit.tokeniser import tokeniser_factory


class SFTDataset(Dataset):
    def __init__(self, data_path: str, sample_rate: int = 16000, torchaudio_backend: Optional[str] = None):
        self.torchaudio_backend = torchaudio_backend
        self.sample_rate = sample_rate
        self.sft_data = []
        with open(data_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                self.sft_data.append(data)

    def __len__(self):
        return len(self.sft_data)

    def load_audio(self, path: str):
        waveform, sample_rate = torchaudio.load(path, backend=self.torchaudio_backend)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        if waveform.dim() == 2:
            waveform = waveform.mean(dim=0)
        return waveform

    def __getitem__(self, idx):
        data = self.sft_data[idx]
        user_audio_path = data['user_audio_path']
        assistant_audio_path = data['assistant_audio_path']

        user_waveform = self.load_audio(user_audio_path)
        assistant_waveform = self.load_audio(assistant_audio_path)

        return (data,
                user_waveform, user_waveform.shape[-1],
                assistant_waveform, assistant_waveform.shape[-1])

    def subsample_data(self, skip: Optional[int], take: Optional[int]):
        if skip is not None:
            self.sft_data = self.sft_data[skip:]
        if take is not None:
            self.sft_data = self.sft_data[:take]


def pad_collate_fn(batch):
    data, user_waveforms, user_l, assistant_waveforms, assistant_l = zip(*batch)
    # Concatenate user and assistant waveforms to process in one batch
    # wav shape: (2*batch_size, max_len)
    wavs = pad_sequence(user_waveforms + assistant_waveforms, batch_first=True)
    return data, wavs, torch.as_tensor(user_l + assistant_l)


@hydra.main(config_name='extract_sft_features', config_path='../config', version_base="1.3")
def extract_features(cfg: DictConfig):
    """
    This function extracts features from an SFT dataset of audio files. It
    accepts a jsonl file with the following format:
    {"user_text": "...", "user_audio_path": "path/to/user.wav", "assistant_text": "...", "assistant_audio_path": "path/to/assistant.wav"}

    Returns jsonl file with the following format:
    {"user_text": "...", "user_audio_path": "...", "user_audio": {"units": [...], "duration": [...]},
     "assistant_text": "...", "assistant_audio_path": "...", "assistant_audio": {"units": [...], "duration": [...]}}
    """
    tokeniser = tokeniser_factory(cfg.tokeniser).to(cfg.device)
    dataset = SFTDataset(cfg.data_path, cfg.sample_rate, cfg.torchaudio_backend)
    dataset.subsample_data(cfg.skip, cfg.take)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=pad_collate_fn)

    with open(cfg.out_path, 'w') as f:
        for data, wavs, lens in tqdm(dataloader):
            wavs, lens = wavs.to(cfg.device), lens.to(cfg.device)
            batch_size = len(data)

            # Extract features for all audio (user + assistant) in one batch
            tokenised = tokeniser.audio_represent(wavs, lens)

            # Split back into user and assistant
            user_tokenised = tokenised[:batch_size]
            assistant_tokenised = tokenised[batch_size:]

            # Write out results
            for i, data_point in enumerate(data):
                data_point['user_audio'] = user_tokenised[i]
                data_point['assistant_audio'] = assistant_tokenised[i]
                f.write(json.dumps(data_point) + '\n')


if __name__ == '__main__':
    extract_features()
