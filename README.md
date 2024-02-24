# pytorch-models

Simple and hackable implementations of some models in PyTorch.

Features:
- Most models are single files, except some standard components like trasnformer, so that they can be easily copy-pasted.
- All models are compatible with `torch.compile()`.
- Weights are loaded from official repos if possible, otherwise from HuggingFace.
- Training is not supported.

## Installation

The only dependency is PyTorch and requests. Follow the official installation instruction [here](https://pytorch.org/), then run:

```bash
pip install git+https://github.com/gau-nernst/pytorch-models.git
```

It is not necessary to install this repo. You can go to whichever model you want and copy-paste it.

## Image

Available models:

- [ViT](https://arxiv.org/abs/2010.11929)
  - [`augreg`](https://arxiv.org/abs/2106.10270) weights: Ti/16, S/32, S/16, B/32, B/16, L/16
  - [`siglip`](https://arxiv.org/abs/2303.15343) weights: B/16 (224, 256, 384, 512), L/16 (256, 384)
  - [`deit3`](https://arxiv.org/abs/2204.07118) weights: S/16, M/16, B/16, L/16, H/16 (layer scale and stochastic depth not implemented)
  - [`dino`](https://arxiv.org/abs/2104.14294) weights: S/16, S/8, B/16, B/8 (stochastic depth not implemented)
  - [`dinov2`](https://arxiv.org/abs/2304.07193) weights: S/14, B/14, L/14 (layer scale and stochastic depth not implemented. input size is 518)
- [MLP-Mixer](https://arxiv.org/abs/2105.01601)
  - `imagenet21k`, `imagenet1k`, or [`sam`](https://arxiv.org/abs/2010.01412) weights: B/16, L/16
  - [`gsam`](https://arxiv.org/abs/2203.08065) weights: S/32, S/16, S/8, B/32, B/16
- [MobileViT](https://arxiv.org/abs/2110.02178): xxs, xs, and s
- [ConvNeXt](https://arxiv.org/abs/2201.03545): tiny, small, base, large, xlarge
- [MaxViT](https://arxiv.org/abs/2204.01697): tiny, small, base, large, xlarge
  - Architecture: MBConv (MobileNet block) + local window attention + grid attention (like MobileViT)
- [DETR](https://arxiv.org/abs/2005.12872)
  - Architecture: ResNet + Transformer Encoder-Decoder (no causal attention in Decoder, so more like Encoder with cross-attention)
  - Weights: R50, R101 (don't support DC5 checkpoints)
  - Don't support image masking (for accurate batch inference of different image sizes)

TODO:

- SAM?

## Text

Available models:

- [BERT](https://arxiv.org/abs/1810.04805) (all HF-compatible BERT and RoBERTa)
  - Google weights: `bert-{base/large}-{uncased/cased}`, `bert-base-multilingual-{uncased/cased}`, `bert-base-chinese`, `bert-large-{uncased/cased}-whole-word-masking`, [mini BERT series](https://huggingface.co/collections/gaunernst/mini-bert-models-656ae9969ced9d5ff5184af0).
  - RoBERTa seris: [`roberta-{base/large}`](https://arxiv.org/abs/1907.11692), [`xlm-roberta-{base/large}`](https://arxiv.org/abs/1911.02116), [`facebook/xlm-roberta-{xl/xxl}`](https://arxiv.org/abs/2105.00572)
  - TODO: add tokenizer
- [T5](https://arxiv.org/pdf/1910.10683) / [Flan-T5](https://arxiv.org/abs/2210.11416) (T5 1.1 + LM-adapted, mT5 + LM-adapted, Flan-T5)
  - `msgpack` is required for loading pre-trained checkpoints. `sentencepiece` is required to use the tokenizer.
- [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (aka GPT-1): L12 H768 (BERT-Base equivalent)
- [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): `gpt2` (L12 H768), `gpt2-medium` (L24 H1024), `gpt2-large` (L36 H1280), `gpt2-xl` (L48 H1600)

TODO:

- Llama
- BART/mBART
- NLLB

## Audio

Available models:

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) / [HuBERT](https://arxiv.org/abs/2106.07447)
- [SEW](https://arxiv.org/abs/2109.06870)
- [data2vec](https://arxiv.org/abs/2202.03555) (audio)
- [EnCodec](https://arxiv.org/abs/2210.13438) (encoder, decoder, and residual vector quantizer)

TODO:

- AST / AudioMAE
- Make models support unbatched inputs
- Support loading EnCodec from AudioCraft

## Audio-to-Text

Available models:

- [Whisper](https://arxiv.org/abs/2212.04356)
  - TODO: add tokenizer
  - TODO: support distilled Whipser

## Image-to-Text

TODO:

- [Donut](https://arxiv.org/abs/2111.15664)
- [Pix2Struct](https://arxiv.org/abs/2210.03347)
- BLIP?

## Usage

### Image

For `ViT` and `MLPMixer` (`imagenet21k`, `imagenet1k`, `sam`, or `gsam`)

```python
import torch
from pytorch_models.image import ViT

model = ViT.from_google("B/16_augreg", pretrained=True)  # also available: siglip
# model = ViT.from_facebook("B/16_deit3", pretrained=True)  # also available: dino, dinov2
outputs = model(torch.randn(1, 3, 224, 224))  # (1, 768)

model.resize_pe(256)  # resize positional embeddings to accept different input size
outputs = model(torch.randn(1, 3, 256, 256))  # (1, 768)
```

For `MobileViT`

```python
import torch
from pytorch_models.image import MobileViT

model = MobileViT.from_apple("xxs", pretrained=True)
outputs = model(torch.randn(1, 3, 256, 256))  # (1, 320)
```

For `ConvNeXt`

```python
import torch
from pytorch_models.image import ConvNeXt

model = ConvNeXt.from_facebook("tiny", pretrained=True)
outputs = model(torch.randn(1, 3, 256, 256))  # (1, 320)
```

For `DETR`

```python
import torch
from pytorch_models.image import DETR, DETRPipeline

m = DETR.from_facebook("resnet50", pretrained=True)  # also available: "resnet101"

img = torch.randn(1, 3, 256, 256)
img = (img - torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

logits, boxes = m(img)
logits  # (1, 100, 92) where 100 is n_queries, 92 is n_classes + 1
boxes  # (1, 100, 4) where 4 is relative box coordinates in cxcywh format (from 0-1)

# to decode outputs
th = 0.7
probs = logits.softmax(-1)[..., :-1]  # skip last class e.g. no-object
keep = probs.amax(-1) >= th  # (n_queries,)

probs = probs[0, keep[0]]  # (n_detections, 92)
boxes = boxes[0, keep[0]]  # (n_detections, 4)

# scale to absolute image pixel coordinates
boxes = boxes * boxes.new_tensor([img.shape[-1], img.shape[-2], img.shape[-1], img.shape[-2]])

# convert to xyxy format
x1 = boxes[..., 0] - boxes[..., 2] * 0.5
y1 = boxes[..., 1] - boxes[..., 3] * 0.5
x2 = boxes[..., 0] + boxes[..., 2] * 0.5
y2 = boxes[..., 1] + boxes[..., 3] * 0.5
boxes = torch.stack([x1, y1, x2, y2], dim=-1)

# simple wrapper, with pre- and post-processing mentioned above
pipeline = DETRPipeline(m, th)

# pass in a list of CHW tensors. output is a list of results for each input image
out = pipeline([torch.randn(3, 256, 256)])[0]
out
# ['remote', 'remote', 'couch', 'cat', 'cat'],
# tensor([[ 3.9384e+01,  6.6804e+01,  1.7851e+02,  1.2038e+02],
#         [ 3.3367e+02,  7.5397e+01,  3.6453e+02,  1.9142e+02],
#         [-1.0672e-01,  1.2313e+00,  6.3982e+02,  4.7406e+02],
#         [ 1.1512e+01,  5.1833e+01,  3.1479e+02,  4.6912e+02],
#         [ 3.4363e+02,  2.4222e+01,  6.3989e+02,  3.6602e+02]]),
# tensor([0.9987, 0.7237, 0.9943, 0.9988, 0.9987])]
```

### Text

For `BERT` and `RoBERTa`

```python
import torch
from pytorch_models.text import BERT

model = BERT.from_hf("bert-base-uncased", pretrained=True)
outputs = model(torch.randint(2000, size=(1, 64)))  # (1, 64, 768)
```

For `T5Model`

```python
import torch
from pytorch_models.text import T5Model, T5Generator

model = T5Model.from_t5x("flan_t5-small", pretrained=True).eval()
tokenizer = T5Model.get_tokenizer("flan_t5-small")

inputs = "Translate to German. What is your name?"
input_ids = tokenizer.Encode(inputs, add_eos=True)
input_ids = torch.tensor(input_ids)

targets = "Welches ist Ihres Namen?"
target_ids = [tokenizer.pad_id()] + tokenizer.Encode(targets, add_eos=True)
target_ids = torch.tensor(target_ids)

# the model supports inputs without batch dim
encoded = model.encode(input_ids)  # call encoder, (n_tokens, d_model)
decoded = model.decode(target_ids, encoded)  # call decoder, (n_tokens, vocab_size)

decoded = model(input_ids, target_ids)  # same as above

# using Generator wrapper (greedy decoding)
generator = T5Generator("flan_t5-small")

prompt = "Translate to German. What is your name?"
answer = generator.generate(prompt)
assert answer == "Welches ist Ihres Namen?"
```

For `GPT` and `GPT2`

```python
import torch
from pytorch_models.text import GPT, GPT2, DecoderGenerator
from transformers import AutoTokenizer

# model = GPT.from_openai(pretrained=True).eval()
# tokenizer = AutoTokenizer.from_pretrained("openai-gpt")

model = GPT2.from_hf("gpt2", pretrained=True).eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

token_ids = torch.tensor(tokenizer.encode("Today is a good day"))  # (n_tokens,)
logits = model(token_ids)  # (n_tokens, vocab_size)

# simple text generator with HF's tokenizer
# set topk=1 for greedy decoding
generator = DecoderGenerator(model, tokenizer)
text = generator.generate("Today is a good day", max_tokens=20, topk=10)
# Sample output: Today is a good day to celebrate the accomplishments of your community.
# We are going to have some really good games coming out next
```

### Audio

For `Wav2Vec2`, `SEW`, and `Data2VecAudio` (weights are from HF):

```python
import torch
from pytorch_models.audio import Wav2Vec2

model = Wav2Vec2.from_hf("facebook/wav2vec2-xls-r-300m", pretrained=True)  # also compatible with HuBERT and MMS weights
outputs = model(torch.randn(2, 16000))  # only supports mono audio. no channel dim.
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

### Audio-to-Text

For `Whisper`:

```python
import torch
from pytorch_models.audio import Whisper, WhisperPreprocessor

preprocessor = WhisperPreprocessor()
model = Whisper.from_openai("tiny.en", pretrained=True)

audio = torch.randn(16000)  # 16kHz audio
melspecs = preprocessor(audio)  # (80, 100)
targets = torch.randint(200, size=(1, 32))
outputs = model(melspecs.unsqueeze(0), targets)  # (1, 32)
```
