import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

class US8KDataset(Dataset):

    def __init__(self, annotations_file_dir, audio_file_dir, transformation, target_sample_rate,  num_samples, device):
        self.annotations = pd.read_csv(annotations_file_dir)
        self.audio = audio_file_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.get_audio_path(index)
        label = self.get_audio_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self.resample(signal, sr)
        signal = self.mix_down(signal)
        signal = self.cut(signal)
        signal = self.right_pad(signal)
        signal = self.transformation(signal)
        return signal, label


    def cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def get_audio_path(self, index):
        fold = f"fold{self.annotations.iloc[index, 5]}"
        path = os.path.join(self.audio, fold, self.annotations.iloc[
            index, 0])
        return path

    def get_audio_label(self, index):
        return self.annotations.iloc[index, 6]


