import timm
import torch

from pytorch_models.image import MLPMixer


def test_forward():
    m = MLPMixer.from_google("S/16")
    m(torch.randn(1, 3, 224, 224))


def test_from_pretrained():
    m = MLPMixer.from_google("B/16_imagenet21k", pretrained=True).eval()
    m_timm = timm.create_model("mixer_b16_224.goog_in21k", pretrained=True, num_classes=0).eval()

    x = torch.randn(1, 3, 224, 224)
    actual = m(x)
    expected = m_timm(x)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
