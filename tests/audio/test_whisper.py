import pytest
import torch
import whisper
from torch import Tensor

from pytorch_models.audio.whisper import WhisperEncoder, WhisperPreprocessor


@pytest.fixture
def x():
    return torch.randn(2, 80, 300)


@torch.no_grad()
def test_forward(x: Tensor):
    m = WhisperEncoder(2, 64)
    m(x)


def test_compile(x: Tensor):
    m = WhisperEncoder(2, 64)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@torch.no_grad()
@pytest.mark.parametrize("variant", ("tiny",))
def test_from_openai(variant: str, x: Tensor):
    x = x.repeat(1, 1, 3000 // x.shape[2])
    m = WhisperEncoder.from_openai("tiny", True).eval()
    m_openai = whisper.load_model(variant).encoder.eval()

    actual = m(x)
    expected = m_openai(x)

    diff = (actual - expected).abs().mean()
    assert diff < 1e-6


def test_preprocess():
    x = torch.randn(16_000)

    actual = WhisperPreprocessor()(x)
    expected = whisper.log_mel_spectrogram(x)

    torch.testing.assert_close(actual, expected)
