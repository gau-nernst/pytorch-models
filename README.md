# pytorch-models

Simple and hackable implementations of some models in PyTorch. Most models are single files, except some standard components like trasnformer, so that they can be easily copy-pasted.

Weights are ported from HF or official repos. Training is not supported.

## Installation

The only dependency is PyTorch and requests. Follow the official installation instruction [here](https://pytorch.org/), then run:

```bash
pip install git+https://github.com/gau-nernst/pytorch-models.git
```

It is not necessary to install this repo. You can go to whichever model you want and copy-paste it.

## Image

Available models:

- [ViT](https://arxiv.org/abs/2010.11929) with [AugReg](https://arxiv.org/abs/2106.10270) and [SigLip](https://arxiv.org/abs/2303.15343) weights

TODO:

- migrate from vision-toolbox
- DETR?
- SAM?

## Text

Available models:

- [BERT](https://arxiv.org/abs/1810.04805) (all HF-compatible BERT)
  - TODO: add tokenizer
- [T5](https://arxiv.org/pdf/1910.10683) / [Flan-T5](https://arxiv.org/abs/2210.11416) (T5 1.1 + LM-adapted, mT5 + LM-adapted, Flan-T5)
  - `msgpack` is required for loading pre-trained checkpoints. `sentencepiece` is required to use the tokenizer.

TODO:

- [RoBERTa](https://arxiv.org/abs/1907.11692)
- [GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Audio

Available models:

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) / [HuBERT](https://arxiv.org/abs/2106.07447)
- [SEW](https://arxiv.org/abs/2109.06870)
- [data2vec](https://arxiv.org/abs/2202.03555) (audio)
- [Whisper](https://arxiv.org/abs/2212.04356) (encoder)
- [EnCodec](https://arxiv.org/abs/2210.13438) (encoder, decoder, and residual vector quantizer)

TODO:

- AST / AudioMAE
- Make models support unbatched inputs
- Support loading EnCodec from AudioCraft

## Audio-to-Text

TODO:

- [Whisper](https://arxiv.org/abs/2212.04356)

## Image-to-Text

TODO:

- [Donut](https://arxiv.org/abs/2111.15664)
- [Pix2Struct](https://arxiv.org/abs/2210.03347)
- BLIP?

## Usage

### Image

```python
import torch
from pytorch_models.image import ViT

model = ViT.from_google("B/16_augreg", pretrained=True)
outputs = model(torch.randn(1, 3, 224, 224))  # (1, 768)

model.resize_pe(256)  # resize positional embeddings to accept different input size
outputs = model(torch.randn(1, 3, 256, 256))  # (1, 768)
```

### Text

For `BERT`

```python
import torch
from pytorch_models.text import BERT

model = BERT.from_hf("bert-base-uncased", pretrained=True)
outputs = model(torch.randint(2000, size=(1, 64)))  # (1, 64, 768)
```

For `T5Model`

```python
import torch
from pytorch_models.text import T5Model

model = T5Model.from_t5x("flan_t5-small", pretrained=True)
tokenizer = T5Model.get_tokenizer("flan_t5-small")

inputs = "Translate to German. What is your name?"
input_ids = tokenizer.Encode(inputs, add_eos=True)
input_ids = torch.tensor(input_ids)

targets = "Welches ist Ihres Namen?"
target_ids = [tokenizer.pad_id()] + tokenizer.Encode(targets, add_eos=True)
target_ids = torch.tensor(target_ids)

# the model supports inputs without batch dim
encoded = model.encode(input_ids)  # call encoder
decoded = model.decode(target_ids, encoded)  # call decoder

decoded = model(input_ids, target_ids)  # same as above
```

### Audio

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
