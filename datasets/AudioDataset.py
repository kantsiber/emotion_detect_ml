import random
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import pandas as pd
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, csv_file, prefix_path, hop_length=256, n_mels=80, n_fft=1024, target_sample_rate=16_000, target_time=3.0):
        self.df = pd.read_csv(csv_file)
        self.prefix_path = prefix_path
        self.hop_length = hop_length
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.target_time = target_time
        self.target_length = int(target_sample_rate * target_time)

        self.emotion_to_label = {
            'neutral': 0,
            'sad': 1,
            'angry': 2,
            'positive': 3
        }

        self.mel_transform = T.MelSpectrogram(
            sample_rate=target_sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_path = self.prefix_path + row['audio_path']
        emotion_str = row['emotion']
        label = self.emotion_to_label[emotion_str]

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.target_sample_rate:
            waveform = F.resample(waveform, sample_rate, self.target_sample_rate)

        current_length = waveform.shape[1]
        if current_length > self.target_length:
            start = waveform.shape[1] - self.target_length
            audio_start = random.randint(0, start)
            waveform = waveform[:, audio_start: audio_start + self.target_length]
        else:
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log(mel_spec + 1e-6)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

        return mel_spec, label