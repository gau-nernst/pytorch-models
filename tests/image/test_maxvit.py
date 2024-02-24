import pytest
import timm
import torch

from pytorch_models.image import MaxViT


@torch.no_grad()
def test_forward():
    m = MaxViT.from_google("tiny")
    m(torch.randn(1, 3, 224, 224))


@torch.no_grad()
def test_compile():
    m = MaxViT.from_google("tiny")
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(torch.randn(1, 3, 224, 224))


@torch.no_grad()
@pytest.mark.parametrize(
    "variant,timm_tag",
    [("tiny", "maxvit_tiny_tf_224.in1k")],
)
def test_from_google(variant, timm_tag):
    m = MaxViT.from_google(variant, pretrained=True).eval()
    m_timm = timm.create_model(timm_tag, pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
