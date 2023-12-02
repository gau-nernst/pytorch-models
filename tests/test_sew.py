import pytest
import torch
from torch import Tensor
from transformers import AutoModel

from pytorch_models.audio.sew import SEW


@pytest.fixture
def x():
    return torch.randn(2, 6400)


def test_forward(x: Tensor):
    m = SEW(2, 64)
    m(x)


def test_compile(x: Tensor):
    m = SEW(2, 64)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@pytest.mark.parametrize("model_id", ("asapp/sew-tiny-100k",))
def test_from_hf(model_id: str, x: Tensor):
    m = SEW.from_hf(model_id, pretrained=True).eval()
    m_hf = AutoModel.from_pretrained(model_id).eval()

    actual = m(x)
    expected = m_hf(x).last_hidden_state

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-6)
