import pytest
import torch
from librosa.core.spectrum import _spectrogram
from librosa.feature import melspectrogram
from torch import Tensor

from pytorch_models.audio.spectrogram import MelSpectrogram, Spectrogram


@pytest.fixture
def x():
    return torch.randn(16_000)


@pytest.fixture
def sample_rate():
    return 16_000


@pytest.mark.parametrize("n_fft,hop_length", ((400, 160),))
def test_spectrogram(x: Tensor, n_fft: int, hop_length: int):
    actual = Spectrogram(n_fft, hop_length)(x)
    expected = _spectrogram(y=x.numpy(), n_fft=n_fft, hop_length=hop_length, power=2, pad_mode="reflect")[0]
    expected = torch.from_numpy(expected)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("n_fft,hop_length,n_mels", ((400, 160, 80),))
def test_mel_spectrogram(x: Tensor, n_fft: int, hop_length: int, n_mels: int, sample_rate: int):
    actual = MelSpectrogram(n_fft, hop_length, n_mels, sample_rate)(x)
    expected = melspectrogram(
        y=x.numpy(), sr=sample_rate, n_fft=n_fft, hop_length=hop_length, pad_mode="reflect", n_mels=n_mels
    )
    expected = torch.from_numpy(expected)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
