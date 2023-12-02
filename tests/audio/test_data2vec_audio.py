import pytest
import torch
from torch import Tensor
from transformers import AutoModel

from pytorch_models.audio.data2vec_audio import Data2VecAudio


@pytest.fixture
def x():
    return torch.randn(2, 6400)


@torch.no_grad()
def test_forward(x: Tensor):
    m = Data2VecAudio(2, 64)
    m(x)


def test_compile(x: Tensor):
    m = Data2VecAudio(2, 64)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@torch.no_grad()
@pytest.mark.parametrize("model_id", ("facebook/data2vec-audio-base",))
def test_from_hf(model_id: str, x: Tensor):
    m = Data2VecAudio.from_hf(model_id, pretrained=True).eval()
    m_hf = AutoModel.from_pretrained(model_id).eval()

    actual = m(x)
    expected = m_hf(x).last_hidden_state

    diff = (actual - expected).abs().mean()
    assert diff < 1e-6
