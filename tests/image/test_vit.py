import pytest
import timm
import torch

from pytorch_models.image import ViT


@torch.no_grad()
def test_forward():
    m = ViT.from_google("Ti/16")
    m(torch.randn(1, 3, 224, 224))


@torch.no_grad()
def test_compile():
    m = ViT.from_google("Ti/16")
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(torch.randn(1, 3, 224, 224))


@torch.no_grad()
def test_resize_pe():
    m = ViT.from_google("Ti/16")
    m(torch.randn(1, 3, 224, 224))
    m.resize_pe(256)
    m(torch.randn(1, 3, 256, 256))


@torch.no_grad()
@pytest.mark.parametrize(
    "model_tag,timm_tag",
    [
        ("Ti/16_augreg", "vit_tiny_patch16_224.augreg_in21k"),
        ("B/16_siglip", "vit_base_patch16_siglip_224"),
    ],
)
def test_from_google(model_tag, timm_tag):
    m = ViT.from_google(model_tag, pretrained=True).eval()
    m_timm = timm.create_model(timm_tag, pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@torch.no_grad()
@pytest.mark.parametrize(
    "model_tag,timm_tag,img_size",
    [
        ("S/16_deit3", "deit3_small_patch16_224.fb_in22k_ft_in1k", 224),
        ("S/16_dino", "vit_small_patch16_224.dino", 224),
        ("S/14_dinov2", "vit_small_patch14_dinov2.lvd142m", 518),
    ],
)
def test_from_facebook(model_tag, timm_tag, img_size):
    m = ViT.from_facebook(model_tag, pretrained=True).eval()
    m_timm = timm.create_model(timm_tag, pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, img_size, img_size)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
