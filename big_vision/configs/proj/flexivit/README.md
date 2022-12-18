# FlexiViT: One Model for All Patch Sizes
*by Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, Filip Pavetic*

## Introduction
We publish all pre-trained FlexiViT models, and configurations for training
those, as well as training logs for one run.

Please read the main [big_vision README](/README.md) to learn how to run
configs, and remember that each config file contains an example invocation in
the top-level comment.

## Pre-trained paper models

Here are the models that we used as backbones in the paper. See Tables in the
appendix of the paper for expected scores at various patch-sizes and on various
datasets.

First, the recommended models we used for all experiments.
Remember that the input is 240px, not 224px:

| Dataset       | Model      | Download link | Notes |
| :---          | :---:      | :---:         | :---: |
| ImageNet-1k   | FlexiViT-L | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k.npz) | 1200ep version |
| ImageNet-1k   | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k.npz) | 1200ep version |
| ImageNet-1k   | FlexiViT-S | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k.npz) | 1200ep version |
| ImageNet-21k  | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_300ep.npz) | 300ep version. 1000ep version below is better but was not used in the paper for fair comparison to baselines. |
| ImageNet-21k  | ViT-B/16   | [link](https://storage.googleapis.com/big_vision/flexivit/vit_b16_i21k_300ep.npz) | Apples-to-apples non-flexi baseline used throughout the paper. |
| ImageNet-21k  | ViT-B/30   | [link](https://storage.googleapis.com/big_vision/flexivit/vit_b30_i21k_300ep.npz) | Apples-to-apples non-flexi baseline used throughout the paper. |

These models can be used directly in our codebase by specifying
`model_name = "proj.flexi.vit"` and `model_init = "FlexiViT-L i1k"` for example.
See the file `models/proj/flexi/vit.py` for more names.

*Important detail:* When further re-using these models with a flexible patch
size, it is recommended to keep the patch-embedding parameter buffer at its
original size, and change patch-size on the fly using pi-resize, as opposed to
changing the parameter buffer's size at load-time.
For re-using the models with a fixed patch size, either way is fine.
(The reason is that it is impossible to chain multiple resizes without loss,
eg doing 32->8->32 does not result in the original weights.)

Second, the list of all released models for completeness:

| Dataset       | Model      | Download link | Notes |
| :---          | :---:      | :---:         | :---: |
| ImageNet-21k  | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_1000ep.npz) | 1000ep version. Should be the best available -B model. |
| ImageNet-21k  | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i21k_90ep.npz) | 90ep version |
| ImageNet-1k   | FlexiViT-L | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_600ep.npz) | 600ep version |
| ImageNet-1k   | FlexiViT-L | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_300ep.npz) | 300ep version |
| ImageNet-1k   | FlexiViT-L | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_l_i1k_90ep.npz) | 90ep version |
| ImageNet-1k   | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_600ep.npz) | 600ep version |
| ImageNet-1k   | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_300ep.npz) | 300ep version |
| ImageNet-1k   | FlexiViT-B | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_b_i1k_90ep.npz) | 90ep version |
| ImageNet-1k   | FlexiViT-S | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_600ep.npz) | 600ep version |
| ImageNet-1k   | FlexiViT-S | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_300ep.npz) | 300ep version |
| ImageNet-1k   | FlexiViT-S | [link](https://storage.googleapis.com/big_vision/flexivit/flexivit_s_i1k_90ep.npz) | 90ep version |

## Results

We provide full training logs for a run with this public code on Cloud that
reproduces the FlexiViT-S 90ep on i1k results:
    - [metrics](https://storage.googleapis.com/big_vision/flexivit/deit3_i1k_s_90ep_12-15_2254/big_vision_metrics.txt)
    - [config](https://storage.googleapis.com/big_vision/flexivit/deit3_i1k_s_90ep_12-15_2254/config.json)
    - or `gs://big_vision/flexivit/deit3_i1k_s_90ep_12-15_2254`.
