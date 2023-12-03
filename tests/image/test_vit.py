import pytest
import timm
import torch

from pytorch_models.image import ViT


def test_forward():
    m = ViT.from_google("Ti/16")
    m(torch.randn(1, 3, 224, 224))


def test_resize_pe():
    m = ViT.from_google("Ti/16")
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


@pytest.mark.parametrize(
    "model_tag,timm_name",
    [
        ("Ti/16_augreg", "vit_tiny_patch16_224.augreg_in21k"),
        ("B/16_siglip", "vit_base_patch16_siglip_224"),
    ],
)
def test_from_google(model_tag, timm_name):
    m = ViT.from_google(model_tag, pretrained=True).eval()
    m_timm = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_from_deit3():
    m = ViT.from_deit3("S/16", pretrained=True).eval()
    m_timm = timm.create_model("deit3_small_patch16_224.fb_in22k_ft_in1k", pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
