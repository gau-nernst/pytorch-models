import math

import torch
from torch import Tensor, nn


class Spectrogram(nn.Module):
    def __init__(self, n_fft: int, hop_length: int) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), False)
        self.window: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return torch.stft(x, self.n_fft, self.hop_length, window=self.window, return_complex=True).abs().square()


def get_mel_filters(n_mels: int, n_fft: int, sample_rate: float) -> Tensor:
    f_max = sample_rate / 2
    mel_max = f_max * 3 / 200 if f_max < 1000 else 15 + 27 * math.log(f_max / 1000, 6.4)

    mel_freqs = torch.linspace(0, mel_max, n_mels + 2)  # linearly spaced in mel-scale
    mel_freqs = torch.where(mel_freqs < 15, mel_freqs * 200 / 3, 1000 * 6.4 ** ((mel_freqs - 15) / 27))
    fft_freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

    mel_diff = mel_freqs.diff()  # (n_mels + 1)
    ramp = mel_freqs.unsqueeze(1) - fft_freqs.unsqueeze(0)  # (n_mels + 2, n_fft / 2 + 1)

    lower = -ramp[:-2] / mel_diff[:-1, None]  # (n_mels, n_fft / 2 + 1)
    upper = ramp[2:] / mel_diff[1:, None]
    filters = lower.minimum(upper).clamp(0)

    filters *= 2 / (mel_freqs[2:, None] - mel_freqs[:-2, None])
    return filters


class MelSpectrogram(Spectrogram):
    def __init__(self, n_fft: int, hop_length: int, n_mels: int, sample_rate: int) -> None:
        super().__init__(n_fft, hop_length)
        self.register_buffer("filters", get_mel_filters(n_mels, n_fft, sample_rate))
        self.filters: Tensor

    def forward(self, x: Tensor) -> Tensor:
        return self.filters @ super().forward(x)
