import pytest
import timm
import torch

from pytorch_models.image import MobileViT


@torch.no_grad()
def test_forward():
    m = MobileViT.from_apple("xxs")
    m(torch.randn(1, 3, 256, 256))


@torch.no_grad()
def test_compile():
    m = MobileViT.from_apple("xxs")
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(torch.randn(1, 3, 256, 256))


@torch.no_grad()
@pytest.mark.parametrize("variant", ["xxs"])
def test_from_google(variant):
    m = MobileViT.from_apple(variant, pretrained=True).eval()
    m_timm = timm.create_model(f"mobilevit_{variant}.cvnets_in1k", pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 256, 256)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
