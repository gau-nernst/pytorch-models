import pytest
import torch
from torch import Tensor
from transformers import AutoModel

from pytorch_models.text import BERT


@pytest.fixture
def x():
    return torch.randint(3, 2000, size=(1, 16))


@torch.no_grad()
def test_forward(x: Tensor):
    m = BERT(2000, 2, 128)
    m(x)


def test_compile(x: Tensor):
    m = BERT(2000, 2, 128)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@torch.no_grad()
@pytest.mark.parametrize("model_tag", ("gaunernst/bert-tiny-uncased", "roberta-base"))
def test_from_hf(model_tag: str, x: Tensor):
    m = BERT.from_hf(model_tag, pretrained=True).eval()
    m_hf = AutoModel.from_pretrained(model_tag).eval()

    actual = m(x)
    expected = m_hf(x).last_hidden_state

    torch.testing.assert_close(actual, expected)


def test_preprocess():
    # TODO: add tokenizer
    pass
