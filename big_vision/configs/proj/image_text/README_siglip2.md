# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features

*by Michael Tschannen\*, Alexey Gritsenko\*, Xiao Wang\*, Muhammad Ferjad Naeem\*, Ibrahim Alabdulmohsin\*,Nikhil Parthasarathy\*, Talfan Evans\*, Lucas Beyer\*, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai\* (\*core contributor)*

[[arxiv]](TODO) [[colab]](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb)

### Summary

We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success of the original [SigLIP](https://arxiv.org/abs/2303.15343). In this second iteration, we extend the original image-text training objective with several prior, independently developed techniques into a unified recipe---this includes captioning-based pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements on localization and dense prediction tasks. We also train variants which support multiple resolutions and preserve the input's native aspect ratio. Finally, we train on a more diverse data-mixture that includes de-biasing techniques, leading to much better multilingual understanding and improved fairness. To allow users to trade off inference cost with performance, we release model checkpoints at four sizes (ViT-B (86M), L (303M), So400m (400M), and g-opt (1B)).

### Checkpoints

Below we provide links to all available checkpoints. The standard (non-NaFlex) checkpoints are compatible with [vit.py](https://github.com/google-research/big_vision/blob/main/big_vision/models/vit.py) and [two_towers.py](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/image_text/two_towers.py) from [SigLIP](https://arxiv.org/abs/2303.15343). The only difference is the vocab size (256k) and the tokenizer (Gemma tokenizer). The NaFlex variant requires a different ViT implementation, [naflex_vit.py](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/image_text/naflex_vit.py), and an adapted image preprocessing, see [ops_naflex.py](https://github.com/google-research/big_vision/blob/main/big_vision/pp/proj/image_text/ops_naflex.py). The [demo colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb) is a good entry point to see how to use the models.

| Model             | ViT       | Res   | HF Transformers             | HF JAX                      | INet 0-shot | COCO T→I | COCO I→T |
|:------------------|:---------:|:-----:|:---------------------------:|:---------------------------:|:-----------:|:--------:|:--------:|
| SigLIP 2          | B/16      | 224   | [link](https://huggingface.co/google/siglip2-base-patch16-224)          | [link](https://huggingface.co/google/siglip2-base-patch16-224-jax)        | 78.2        | 52.1     | 68.9     |
| SigLIP 2          | B/16      | 256   | [link](https://huggingface.co/google/siglip2-base-patch16-256)          | [link](https://huggingface.co/google/siglip2-base-patch16-256-jax)        | 79.1        | 53.2     | 69.7     |
| SigLIP 2          | B/16      | 384   | [link](https://huggingface.co/google/siglip2-base-patch16-384)          | [link](https://huggingface.co/google/siglip2-base-patch16-384-jax)        | 80.6        | 54.6     | 71.4     |
| SigLIP 2          | B/16      | 512   | [link](https://huggingface.co/google/siglip2-base-patch16-512)          | [link](https://huggingface.co/google/siglip2-base-patch16-512-jax)        | 81.2        | 55.2     | 71.2     |
| SigLIP 2          | B/32      | 256   | [link](https://huggingface.co/google/siglip2-base-patch32-256)          | [link](https://huggingface.co/google/siglip2-base-patch32-256-jax)        | 74.0        | 47.2     | 63.7     |
| SigLIP 2          | L/16      | 256   | [link](https://huggingface.co/google/siglip2-large-patch16-256)         | [link](https://huggingface.co/google/siglip2-large-patch16-256-jax)       | 82.5        | 54.7     | 71.5     |
| SigLIP 2          | L/16      | 384   | [link](https://huggingface.co/google/siglip2-large-patch16-384)         | [link](https://huggingface.co/google/siglip2-large-patch16-384-jax)       | 83.1        | 55.3     | 71.4     |
| SigLIP 2          | L/16      | 512   | [link](https://huggingface.co/google/siglip2-large-patch16-512)         | [link](https://huggingface.co/google/siglip2-large-patch16-512-jax)       | 83.5        | 55.2     | 72.1     |
| SigLIP 2          | So400m/14 | 224   | [link](https://huggingface.co/google/siglip2-so400m-patch14-224)        | [link](https://huggingface.co/google/siglip2-so400m-patch14-224-jax)      | 83.2        | 55.1     | 71.5     |
| SigLIP 2          | So400m/14 | 384   | [link](https://huggingface.co/google/siglip2-so400m-patch14-384)        | [link](https://huggingface.co/google/siglip2-so400m-patch14-384-jax)      | 84.1        | 55.8     | 71.7     |
| SigLIP 2          | So400m/16 | 256   | [link](https://huggingface.co/google/siglip2-so400m-patch16-256)        | [link](https://huggingface.co/google/siglip2-so400m-patch16-256-jax)      | 83.4        | 55.4     | 71.5     |
| SigLIP 2          | So400m/16 | 384   | [link](https://huggingface.co/google/siglip2-so400m-patch16-384)        | [link](https://huggingface.co/google/siglip2-so400m-patch16-384-jax)      | 84.1        | 56.0     | 71.2     |
| SigLIP 2          | So400m/16 | 512   | [link](https://huggingface.co/google/siglip2-so400m-patch16-512)        | [link](https://huggingface.co/google/siglip2-so400m-patch16-512)          | 84.3        | 56.0     | 71.3     |
| SigLIP 2          | g-opt/16  | 256   | [link](https://huggingface.co/google/siglip2-giant-opt-patch16-256)     | [link](https://huggingface.co/google/siglip2-giant-opt-patch16-256-jax)   | 84.5        | 55.7     | 72.5     |
| SigLIP 2          | g-opt/16  | 384   | [link](https://huggingface.co/google/siglip2-giant-opt-patch16-384)     | [link](https://huggingface.co/google/siglip2-giant-opt-patch16-384-jax)   | 85.0        | 56.1     | 72.8     |
| SigLIP 2 (NaFlex) | B/16      | var.  | [link](https://huggingface.co/google/siglip2-base-patch16-naflex)        | [link](https://huggingface.co/google/siglip2-base-patch16-naflex-jax)     | 78.5        | 51.1     | 67.3     |
| SigLIP 2 (NaFlex) | So400m/16 | var.  | [link](https://huggingface.co/google/siglip2-so400m-patch16-naflex)      | [link](https://huggingface.co/google/siglip2-so400m-patch16-naflex-jax)   | 83.5        | 55.1     | 71.2     |

*The NaFlex results are for sequence length 256.*

\
\
Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0); you may not use this file except in compliance with the Apache 2.0 license. You may obtain a copy of the Apache 2.0 license at: https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0 International License (CC-BY). You may obtain a copy of the CC-BY license at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and materials distributed here under the Apache 2.0 or CC-BY licenses are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the licenses for the specific language governing permissions and limitations under those licenses.

This is not an official Google product.
