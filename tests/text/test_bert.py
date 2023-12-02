import pytest
import torch
from torch import Tensor
from transformers import BertModel

from pytorch_models.text import BERT


@pytest.fixture
def x():
    return torch.randint(2000, size=(1, 16))


def test_forward(x: Tensor):
    m = BERT(2000, 2, 16)
    m(x)


def test_compile(x: Tensor):
    m = BERT(2000, 2, 16)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@pytest.mark.parametrize("model_tag", ("gaunernst/bert-tiny-uncased",))
def test_from_hf(model_tag: str, x: Tensor):
    m = BERT.from_hf(model_tag, pretrained=True).eval()
    m_hf = BertModel.from_pretrained(model_tag).eval()

    actual = m(x)
    expected = m_hf(x).last_hidden_states

    torch.testing.assert_close(actual, expected)
    # diff = (actual - expected).abs().mean()
    # assert diff < 1e-6


def test_preprocess():
    # TODO: add tokenizer
    pass
