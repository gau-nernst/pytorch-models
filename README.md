# pytorch-models

Simple implementation of some models in PyTorch. Weights are ported from HF or official repos. Training is not supported.

## Installation

The only dependency is PyTorch. Follow the official installation instruction [here](https://pytorch.org/), then run:

```bash
pip install git+https://github.com/gau-nernst/pytorch-models.git
```

## Audio

Available models:

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) / [HuBERT](https://arxiv.org/abs/2106.07447)
- [SEW](https://arxiv.org/abs/2109.06870)
- [data2vec](https://arxiv.org/abs/2202.03555) (audio)
- [Whisper](https://arxiv.org/abs/2212.04356) (encoder)
- [EnCodec](https://arxiv.org/abs/2210.13438) (encoder, decoder, and residual vector quantizer)

TODO:

- AST / AudioMAE

### Usage

For `Wav2Vec2`, `SEW`, and `Data2VecAudio` (weights are from HF):

```python
import torch
from pytorch_models.audio import Wav2Vec2

model = Wav2Vec2.from_hf("facebook/wav2vec2-xls-r-300m", pretrained=True)  # also compatible with HuBERT and MMS weights
outputs = model(torch.randn(2, 16000))  # only supports mono audio. no channel dim.
```

For `WhisperEncoder` (weights are from https://github.com/openai/whisper):

```python
import torch
from pytorch_models.audio import WhisperEncoder, WhisperPreprocessor

preprocessor = WhisperPreprocessor()
model = WhisperEncoder.from_openai("tiny.en", pretrained=True)

# TODO: add batch support for WhisperPreprocessor
melspecs = preprocessor(torch.randn(16000))  # (80, 100)
outputs = model(melspecs.unsqueeze(0))
```

For `EnCodec` (weights are from https://github.com/facebookresearch/encodec):

```python
import torch
from pytorch_models.audio import EnCodec

model = EnCodec.from_facebook("24khz", pretrained=True)  # 48khz is also available

# 24khz model accepts mono audio. channel dim is required.
# 48khz model accepts stereo audio.
audio = torch.randn(2, 1, 16000)

# codes has shape (2, 32, 50), where 32 is the number of codebooks.
# scale is None for 24khz model, but it will be a re-scaling factor for 48khz model
codes, scale = model.encode(audio)

# reconstruct audio from codes and scale
reconstructed = model.decode(codes, scale)
```
