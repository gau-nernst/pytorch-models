import pytest
import torch
from torch import Tensor
from transformers import AutoTokenizer, OpenAIGPTLMHeadModel

from pytorch_models.text import GPT, DecoderGenerator


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

    torch.testing.assert_close(actual, expected)


def test_generate():
    prompt = "Today is a good day"

    m = GPT.from_openai(pretrained=True).eval()
    tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

    generator = DecoderGenerator(m, tokenizer)
    actual = generator.generate(prompt, max_tokens=10, topk=1)

    m_hf = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    tokens = m_hf.generate(tokens, max_new_tokens=10).squeeze(0)
    expected = tokenizer.decode(tokens)

    assert actual == expected
