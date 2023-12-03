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
def test_from_pretrained(model_tag, timm_name):
    m = ViT.from_google(model_tag, pretrained=True).eval()
    x = torch.randn(1, 3, 224, 224)
    out = m(x)

    m_timm = timm.create_model(timm_name, pretrained=True, num_classes=0).eval()
    out_timm = m_timm(x)

    torch.testing.assert_close(out, out_timm, rtol=2e-5, atol=2e-5)
