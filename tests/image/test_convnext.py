import pytest
import timm
import torch

from pytorch_models.image import ConvNeXt


@torch.no_grad()
def test_forward():
    m = ConvNeXt.from_facebook("tiny")
    m(torch.randn(1, 3, 224, 224))


@torch.no_grad()
def test_compile():
    m = ConvNeXt.from_facebook("tiny")
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(torch.randn(1, 3, 224, 224))


@torch.no_grad()
@pytest.mark.parametrize("variant", ["tiny", "base"])
def test_from_facebook(variant):
    m = ConvNeXt.from_facebook(variant, pretrained=True).eval()
    m_timm = timm.create_model(f"convnext_{variant}.fb_in22k", pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
