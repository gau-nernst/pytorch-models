import pytest
import torch

from pytorch_models.image import DETR


@torch.no_grad()
def test_forward():
    m = DETR.from_facebook("resnet50")
    m(torch.randn(1, 3, 224, 224))


@torch.no_grad()
def test_compile():
    m = DETR.from_facebook("resnet50")
    m_compiled = torch.compile(m, fullgraph=True)
    m_compiled(torch.randn(1, 3, 224, 224))


@torch.no_grad()
@pytest.mark.parametrize("model_tag,hub_name", [("resnet50", "detr_resnet50")])
def test_from_facebook(model_tag, hub_name):
    m = DETR.from_facebook(model_tag, pretrained=True).eval()
    m_fb = torch.hub.load("facebookresearch/detr:main", hub_name, pretrained=True).eval()

    x = torch.randn(1, 3, 224, 224)
    logits, bboxes = m(x)
    expected = m_fb(x)

    torch.testing.assert_close(logits, expected["pred_logits"], atol=3e-5, rtol=2e-5)
    torch.testing.assert_close(bboxes.sigmoid(), expected["pred_boxes"])
