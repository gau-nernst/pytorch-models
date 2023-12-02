import pytest
import torch
from encodec.model import EncodecModel
from encodec.modules.seanet import SEANetDecoder, SEANetEncoder

from pytorch_models.audio.encodec import EnCodec, EnCodecDecoder, EnCodecEncoder


encoder_decoder = (
    (EnCodecEncoder, (2, 1, 3200)),
    (EnCodecDecoder, (2, 128, 10)),
)


@torch.no_grad()
@pytest.mark.parametrize("cls,x_shape", encoder_decoder)
def test_forward(cls, x_shape: tuple[int, ...]):
    m = cls(1)
    x = torch.randn(x_shape)
    m(x)


@pytest.mark.parametrize("norm_type,causal", (("weight_norm", True), ("time_group_norm", False)))
@pytest.mark.parametrize("cls,x_shape", encoder_decoder)
def test_compile(cls, x_shape: tuple[int, ...], norm_type: str, causal: bool):
    m = cls(1, norm_type=norm_type, causal=causal)
    x = torch.randn(x_shape)

    # there will be a graph break at LSTM
    m_compiled = torch.compile(m)
    m_compiled(x).sum().backward()


@torch.no_grad()
@pytest.mark.parametrize("norm_type,causal", (("weight_norm", True), ("time_group_norm", False)))
@pytest.mark.parametrize("cls,x_shape", encoder_decoder)
def test_load_facebook_state_dict(cls, x_shape: tuple[int, ...], norm_type: str, causal: bool):
    cls_fb = {EnCodecEncoder: SEANetEncoder, EnCodecDecoder: SEANetDecoder}[cls]

    m = cls(1, norm_type=norm_type, causal=causal)
    m_fb = cls_fb(1, norm=norm_type, causal=causal)

    m.load_facebook_state_dict(m_fb.state_dict())

    x = torch.randn(x_shape)
    actual = m(x)
    expected = m_fb(x)

    torch.testing.assert_close(actual, expected)


@torch.no_grad()
@pytest.mark.parametrize("variant,audio_channels", (("24khz", 1), ("48khz", 2)))
def test_from_facebook(variant: str, audio_channels: int):
    method_fb = {"24khz": EncodecModel.encodec_model_24khz, "48khz": EncodecModel.encodec_model_48khz}[variant]

    m = EnCodec.from_facebook(variant, True)
    m_fb = method_fb()

    x = torch.randn(2, audio_channels, 3200)

    actual, scale_actual = m.encode(x)
    expected, scale_expected = m_fb._encode_frame(x)
    torch.testing.assert_close(actual, expected)

    if scale_actual is not None and scale_expected is not None:
        torch.testing.assert_close(scale_actual.view_as(scale_expected), scale_expected)

    actual = m.decode(actual, scale_actual)
    expected = m_fb._decode_frame((expected, scale_expected))
    torch.testing.assert_close(actual, expected)
