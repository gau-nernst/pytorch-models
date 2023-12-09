import pytest
import torch
from torch import Tensor
from transformers import OpenAIGPTLMHeadModel

from pytorch_models.text import GPT


@pytest.fixture
def x():
    return torch.randint(3, 2000, size=(1, 16))


@torch.no_grad()
def test_forward(x: Tensor):
    m = GPT(2, 128)
    m(x)


def test_compile(x: Tensor):
    m = GPT(2, 128)
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(x).sum().backward()


@torch.no_grad()
def test_from_hf(x: Tensor):
    m = GPT.from_openai(pretrained=True).eval()
    m_hf = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").eval()

    actual = m(x)
    expected = m_hf(x).logits

    torch.testing.assert_close(actual[..., : expected.shape[-1]], expected)


def test_preprocess():
    # TODO: add tokenizer
    pass
