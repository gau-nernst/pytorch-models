from functools import partial

import pytest
import torch
import whisper

from pytorch_models.audio2text.whisper import Whisper, WhisperDecoder, WhisperEncoder, WhisperPreprocessor


vocab_size = 100
n_layers = 2
d_model = 64
factory_inputs = [
    (WhisperEncoder, [torch.randn(2, 80, 16)]),
    (partial(WhisperDecoder, vocab_size), [torch.randint(vocab_size, size=(2, 32)), torch.randn(2, 16, d_model)]),
    (partial(Whisper, vocab_size), [torch.randn(2, 80, 16), torch.randint(vocab_size, size=(2, 32))]),
]


@torch.no_grad()
@pytest.mark.parametrize("factory,inputs", factory_inputs)
def test_forward(factory, inputs):
    m = factory(n_layers, d_model)
    m(*inputs)


@pytest.mark.parametrize("factory,inputs", factory_inputs)
def test_compile(factory, inputs):
    m = factory(n_layers, d_model)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(*inputs).sum().backward()


@torch.no_grad()
@pytest.mark.parametrize("model_tag", ("tiny", "tiny.en"))
def test_from_openai(model_tag: str):
    m = Whisper.from_openai(model_tag, pretrained=True).eval()
    m_openai = whisper.load_model(model_tag).eval()

    melspec = torch.randn(1, 80, 3000)
    targets = torch.randint(200, size=(1, 32))
    actual = m(melspec, targets)
    expected = m_openai(melspec, targets)

    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=5e-5)


def test_preprocess():
    x = torch.randn(16_000)

    actual = WhisperPreprocessor()(x)
    expected = whisper.log_mel_spectrogram(x)

    torch.testing.assert_close(actual, expected)
