import pytest
import torch
from torch import Tensor
from transformers import AutoModel

from pytorch_models.audio.wav2vec2 import Wav2Vec2


@pytest.fixture
def x():
    return torch.randn(2, 6400)


def test_forward(x: Tensor):
    m = Wav2Vec2(2, 64)
    m(x)


@pytest.mark.parametrize("stem_legacy", (False, True))
def test_compile(stem_legacy: bool, x: Tensor):
    m = Wav2Vec2(2, 64, stem_legacy=stem_legacy)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@pytest.mark.parametrize(
    "model_id",
    (
        "facebook/wav2vec2-base",
        "facebook/wav2vec2-xls-r-300m",
        "facebook/hubert-base-ls960",
    ),
)
def test_from_hf(model_id: str, x: Tensor):
    m = Wav2Vec2.from_hf(model_id, pretrained=True).eval()
    m_hf = AutoModel.from_pretrained(model_id).eval()

    actual = m(x)
    expected = m_hf(x).last_hidden_state

    # not sure why wav2vec2-base has much larger diff
    tolerance = 4e-6 if model_id == "facebook/wav2vec2-base" else 1e-6
    diff = (actual - expected).abs().mean()
    assert diff < tolerance
