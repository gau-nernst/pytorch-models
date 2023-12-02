from functools import partial

import pytest
import torch
from transformers import T5ForConditionalGeneration

# from generate import T5Generator
from pytorch_models.text import T5Model, T5Stack


dim, n_heads, n_layers, mlp_dim = 512, 6, 8, 1024  # small version
cls_inputs = [
    (partial(T5Stack, decoder=False), (torch.randn(2, 64, dim),)),
    (partial(T5Stack, decoder=True), (torch.randn(2, 64, dim), torch.randn(2, 32, dim))),
    (T5Model, (torch.randint(1000, size=(2, 64)), torch.randint(1000, size=(2, 32)))),
]


@torch.no_grad()
@pytest.mark.parametrize("cls,inputs", cls_inputs)
def test_forward(cls, inputs):
    m = cls(dim, n_heads, n_layers, mlp_dim)
    m(*inputs)


@torch.no_grad()
@pytest.mark.parametrize("cls,inputs", cls_inputs)
def test_forward_non_batched(cls, inputs):
    m = cls(dim, n_heads, n_layers, mlp_dim)
    m(*[x[0] for x in inputs])


@pytest.mark.parametrize("cls,inputs", cls_inputs)
def test_encoder_compile(cls, inputs):
    m = cls(dim, n_heads, n_layers, mlp_dim)
    torch.compile(m)(*inputs).sum().backward()


@pytest.mark.parametrize(
    "checkpoint,vocab_size",
    [("t5_1_1", 32128), ("flan_t5", 32128)],
)
def test_pretrained(checkpoint, vocab_size):
    T5Model.create_model("small", checkpoint=checkpoint, vocab_size=vocab_size)


@torch.no_grad()
def test_against_hf():
    m = T5Model.create_model("small", checkpoint="t5_1_1")
    m_hf = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")

    tokenizer = T5Model.get_tokenizer("t5_1_1")
    x = torch.tensor(tokenizer.Encode("Good morning", add_eos=True)).view(1, -1)
    y = torch.tensor([tokenizer.pad_id()] + tokenizer.Encode("Chao buoi sang")).view(1, -1)

    actual = m(x, y)
    expected = m_hf(input_ids=x, decoder_input_ids=y)[0]

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


# def test_generator():
#     generator = T5Generator("small", checkpoint="flan_t5")
#     prompt = "Translate to German. What is your name?"

#     answer = generator.generate(prompt)
#     assert answer == "Welches ist Ihres Namen?"