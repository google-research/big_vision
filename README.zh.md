# Big Vision

[English](README.md) | [简体中文](README.zh.md)

本代码库专为使用 [Cloud TPU VM](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms) 或 GPU 机器训练大规模视觉模型而设计。它基于 [Jax](https://github.com/jax-ml/jax)/[Flax](https://github.com/google/flax) 库，并使用 [tf.data](https://www.tensorflow.org/guide/data) 与 [TensorFlow Datasets](https://www.tensorflow.org/datasets) 构建可扩展且可复现的输入流水线。

开源本代码库有两个主要目的：
1. 发布基于此代码库开发的研究项目的代码（见下文列表）。
2. 为在 GPU 机器和 Google Cloud TPU 上运行大规模视觉实验提供强大的起点，可无缝开箱即用，从单 TPU 核心扩展到拥有高达 2048 个 TPU 核心的分布式设置。

`big_vision` 旨在支持 Google 内部的研究项目。我们不太可能处理功能请求（feature requests）或接受外部贡献，除非经过预先批准（请先在 Issue 中询问）。如需完全专注于迁移学习且维护良好的代码库，另请参阅 [vision_transformer](https://github.com/google-research/vision_transformer)。

请注意，`big_vision` 是一个相当动态的代码库；虽然我们打算始终保持核心代码的功能完整，但无法保证及时更新位于 `.../proj/...` 子文件夹中的项目专属代码。不过，我们提供了一个[表格](#各项目特定-commit-记录)，记录了特定项目已知可正常运行的最后提交。

以下研究项目最初均在 `big_vision` 代码库中进行开发：

### 架构研究

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)，作者：
  Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
  Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
  Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby*
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560)，作者：
  Xiaohua Zhai*, Alexander Kolesnikov*, Neil Houlsby, and Lucas Beyer*\
  资源：[config](big_vision/configs/proj/scaling_laws/train_vit_g.py)。
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270)，作者：
  Andreas Steiner*, Alexander Kolesnikov*, Xiaohua Zhai*, Ross Wightman,
  Jakob Uszkoreit, and Lucas Beyer*
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601)，作者：
  Ilya Tolstikhin*, Neil Houlsby*, Alexander Kolesnikov*, Lucas Beyer*,
  Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
  Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy\
  资源：[config](big_vision/configs/mlp_mixer_i1k.py)。
- [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580)，作者：
  Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov\
  资源：[config](big_vision/configs/vit_s16_i1k.py)
- [UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes](https://arxiv.org/abs/2205.10337)，作者：
  Alexander Kolesnikov^*, André Susano Pinto^*, Lucas Beyer*, Xiaohua Zhai*, Jeremiah Harmsen*, Neil Houlsby*\
  资源：[readme](big_vision/configs/proj/uvim/README.md)、[configs](big_vision/configs/proj/uvim)、[colabs](big_vision/configs/proj/uvim)。
- [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013)，作者：
  Lucas Beyer*, Pavel Izmailov*, Alexander Kolesnikov*, Mathilde Caron*, Simon
  Kornblith*, Xiaohua Zhai*, Matthias Minderer*, Michael Tschannen*, Ibrahim
  Alabdulmohsin*, Filip Pavetic*\
  资源：[readme](big_vision/configs/proj/flexivit/README.md)、[configs](big_vision/configs/proj/flexivit)。
- [Dual PatchNorm](https://arxiv.org/abs/2302.01327)，作者：Manoj Kumar, Mostafa Dehghani, Neil Houlsby。
- [Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design](https://arxiv.org/abs/2305.13035)，作者：
  Ibrahim Alabdulmohsin*, Xiaohua Zhai*, Alexander Kolesnikov, Lucas Beyer*。
- (部分) [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442)，作者：
  Mostafa Dehghani*, Josip Djolonga*, Basil Mustafa*, Piotr Padlewski*, Jonathan Heek*, *wow many middle authors*, Neil Houlsby*。
- (部分) [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505)，作者：
  Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen。
- [GIVT: Generative Infinite-Vocabulary Transformers](https://arxiv.org/abs/2312.02116)，作者：
  Michael Tschannen, Cian Eastwood, Fabian Mentzer。\
  资源：[readme](big_vision/configs/proj/givt/README.md)、[config](big_vision/configs/proj/givt/givt_imagenet2012.py)、[colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/givt/givt_demo_colab.ipynb)。
- [Unified Auto-Encoding with Masked Diffusion](https://arxiv.org/abs/2406.17688)，作者：
  Philippe Hansen-Estruch, Sriram Vishwanath, Amy Zhang, Manan Tomar。
- [Jet: A Modern Transformer-Based Normalizing Flow](https://arxiv.org/abs/2412.15129)，作者：
  Alexander Kolesnikov*, André Susano Pinto*, Michael Tschannen*，[configs](big_vision/configs/proj/jet)
- [JetFormer: An autoregressive generative model of raw images and text](https://arxiv.org/abs/2411.19722)，作者：
  Michael Tschannen*, André Susano Pinto*, Alexander Kolesnikov*。[configs](big_vision/configs/proj/jetformer)。

### 多模态研究

- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991)，作者：
  Xiaohua Zhai*, Xiao Wang*, Basil Mustafa*, Andreas Steiner*, Daniel Keysers,
  Alexander Kolesnikov, and Lucas Beyer*\
  资源：[trainer](big_vision/trainers/proj/image_text/contrastive.py)、[config](big_vision/configs/proj/image_text/lit_coco.py)、[colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/lit.ipynb)。
- [CLIPPO: Image-and-Language Understanding from Pixels Only](https://arxiv.org/abs/2212.08045)，作者：
  Michael Tschannen, Basil Mustafa, Neil Houlsby\
  资源：[readme](big_vision/configs/proj/clippo/README.md)、[config](big_vision/configs/proj/clippo/train_clippo.py)、[colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb)。
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)，作者：
  Xiaohua Zhai*, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer*\
  资源：[colab 与模型](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb)，代码开发中（TODO）。
- [A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision](https://arxiv.org/abs/2303.17376)，作者：
  Lucas Beyer*, Bo Wan*, Gagan Madan*, Filip Pavetic*, Andreas Steiner*, Alexander Kolesnikov, André Susano Pinto, Emanuele Bugliarello, Xiao Wang, Qihang Yu, Liang-Chieh Chen, Xiaohua Zhai*。
- [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915)，作者：
  Michael Tschannen*, Manoj Kumar*, Andreas Steiner*, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*。\
  资源：[readme](big_vision/configs/proj/cappa/README.md)、[config](big_vision/configs/proj/cappa/pretrain.py)、[model](big_vision/models/proj/cappa/cappa.py)。
- [Three Towers: Flexible Contrastive Learning with Pretrained Image Models](https://arxiv.org/abs/2305.16999)，作者：Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Efi Kokiopoulou。
- (部分) [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794)，作者：Xi Chen, Xiao Wang, Soravit Changpinyo, *wow so many middle authors*, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut。
- (部分) [PaLI-3 Vision Language Models: Smaller, Faster, Stronger](https://arxiv.org/abs/2310.09199)，作者：Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut。
- [LocCa](https://arxiv.org/abs/2403.19596)，作者：
  Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, Xiaohua Zhai。
- [PaliGemma](https://arxiv.org/abs/2407.07726)、
  [PaliGemma 2](https://arxiv.org/abs/2412.03555)，作者：*wow many authors*。\
  资源：[readme](big_vision/configs/proj/paligemma/README.md)、
    [model](big_vision/models/proj/paligemma/paligemma.py)、
    [transfer configs](big_vision/configs/proj/paligemma/transfers)、
    [datasets](big_vision/datasets)、
    [CountBenchQA](big_vision/datasets/countbenchqa/data/countbench_paired_questions.json)。
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786)，作者：*wow many authors*。\
  资源：[readme (包含检查点)](big_vision/configs/proj/image_text/README_siglip2.md)、[colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb)。

### 训练技术

- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237)，作者：
  Lucas Beyer*, Xiaohua Zhai*, Amélie Royer*, Larisa Markeeva*, Rohan Anil,
  and Alexander Kolesnikov*\
  资源：[README](big_vision/configs/proj/distill/README.md)、[trainer](big_vision/trainers/proj/distill/distill.py)、[colab](https://colab.research.google.com/drive/1nMykzUzsfQ_uAxfj3k35DYsATnG_knPl?usp=sharing)。
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412)，作者：
  Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065)，作者：Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha Dvornek, Sekhar Tatikonda, James Duncan and Ting Liu \
  资源：[trainer](big_vision/trainers/proj/gsam/gsam.py)、[config](big_vision/configs/proj/gsam/vit_i1k_gsam_no_aug.py)、[复现结果](https://github.com/google-research/big_vision/pull/8#pullrequestreview-1078557411)
- [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242)，作者：
  André Susano Pinto*, Alexander Kolesnikov*, Yuge Shi, Lucas Beyer, Xiaohua Zhai。
- (部分) [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760)，作者：
  Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, Jascha Sohl-Dickstein。

### 其他研究

- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159)，作者：
  Lucas Beyer*, Olivier J. Hénaff*, Alexander Kolesnikov*, Xiaohua Zhai*, Aäron van den Oord*。
- [No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models](https://arxiv.org/abs/2405.13777)，作者：
  Angéline Pouget, Lucas Beyer, Emanuele Bugliarello, Xiao Wang, Andreas Peter Steiner, Xiaohua Zhai, Ibrahim Alabdulmohsin。

# 代码库高层组织与核心原则概览

主要入口点是训练器（trainer）模块，通常负责处理创建模型和优化器、加载数据、保存检查点以及在循环中训练/评估模型的所有模板代码。我们在根目录下提供了规范的训练器 `train.py`。通常，`big_vision` 中的各个项目会分叉并定制该训练器。

所有模型、评估器和预处理操作均存放在对应的子目录中，通常可以在不同项目之间复用。我们鼓励在这些目录内使用兼容的 API 以促进复用性，但并不严格强制，因为单个项目可能需要引入自定义 API。

我们拥有强大的配置系统，配置文件位于 `configs/` 目录中。自定义训练器和模块可以直接扩展/修改配置项。

项目专属代码位于 `.../proj/...` 命名空间下。项目专属代码并不总能与核心 `big_vision` 库保持同步。我们在下文提供了每个项目的[已知最后正常提交记录](#各项目特定-commit-记录)，在这些 commit 上项目代码预期能正常运行。

训练作业对中断具有鲁棒性，并将从最后保存的检查点无缝恢复（前提是用户提供了正确的 `--workdir` 路径）。

每个配置文件顶部都包含带有 `COMMAND` 代码片段的注释，用于指导如何运行它，并提示预期的运行时间与结果。更多详情见下文；但一般而言，在 GPU 机器上运行涉及调用 `python -m COMMAND`，而在 TPU（包括多主机）上运行则涉及：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all \
  --command "bash big_vision/run_tpu.sh COMMAND"
```

更多关于如何在 GPU 机器或 Google Cloud TPU 上运行 `big_vision` 代码的说明请参见下文。

默认情况下，我们会写入检查点和日志文件。日志文件为 JSON 对象列表，我们提供了一个简短直接的[读取和展示日志与检查点的示例 Colab](https://colab.research.google.com/drive/1R_lvV542WUp8Q2y8sbyooZOGCplkn7KI?usp=sharing)。

# 当前与未来内容

首个版本包含了在 Cloud TPU VM 上大规模预训练、迁移和评估分类模型的核心部分。

自此我们添加了以下关键功能与项目：
- 像 LiT 与 CLIP 一样的对比图像-文本（Image-Text）模型训练与评估。
- 耐心且一致的知识蒸馏。
- 扩展 ViT。
- MLP-Mixer。
- UViM。

我们计划在不久的将来发布的功能与项目（排名不分先后）：
- TFDS 中的 ImageNet-21k。
- 加载我们在论文中使用的各类公开模型（NFNet、MoCov3、DINO）。
- 内存高效的 Polyak 平均实现。
- 高级 JAX 计算与内存性能分析。我们内部正在使用专用工具，但最终可能会添加对公开可用工具的支持。

我们将在此继续发布未来在 `big_vision` 内开发的出版物代码。

### 不包含的内容

以下内容存在于该代码库的内部版本中，目前没有公开发布计划：
- 针对质量与速度的常规回归测试。它们重度依赖内部基础设施。
- 实验的高级日志记录、监控与绘图。这同样高度依赖内部基础设施。不过，我们对此类建议持开放态度，未来可能会添加一些，特别是如果以自包含方式实现的话。
- 尚未发表的正在进行中的研究项目。

# GPU 环境配置

我们首先介绍如何在（本地）GPU 机器上配置并运行 `big_vision`，然后介绍 Cloud TPU 的配置。请注意，（本地）GPU 配置的数据准备步骤在很大程度上可复用于 Cloud TPU 配置。虽然为简明起见指南中省略了此步骤，但我们强烈建议在安装 Python 依赖项时使用[虚拟环境（virtual environment）](https://docs.python.org/3/library/venv.html)。

## 配置 Python 软件包

第一步是检出 `big_vision` 并安装相关的 Python 依赖项：

```
git clone https://github.com/google-research/big_vision
cd big_vision/
pip3 install --upgrade pip
pip3 install -r big_vision/requirements.txt
```

最新版本的 `jax` 库可以通过以下方式获取：

```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

根据您机器上安装的 CUDA 和 cuDNN 库版本，您可能需要不同的 `jax` 安装包。有关更多信息，请参考 [JAX 官方文档](https://github.com/jax-ml/jax#pip-installation-gpu-cuda)。

## 准备 TFDS 数据

为了统一且可复现地访问标准数据集，我们选择使用 `tensorflow_datasets` (`tfds`) 库。它要求将每个数据集下载、预处理并存储在硬盘上（如果使用 "Google Cloud"，最好存储在 "GCP bucket" 中）。

许多数据集在首次使用时可以完全自动下载和预处理。然而，我们特意禁用了此功能，并建议在首次运行之前单独进行数据集准备步骤。如果出现问题，这将使调试更加容易，而且某些数据集（如 `imagenet2012`）需要手动下载数据。

大部分数据集，例如 `cifar100`、`oxford_iiit_pet` 或 `imagenet_v2`，可以通过运行以下命令全自动下载并准备：

```
cd big_vision/
python3 -m big_vision.tools.download_tfds_datasets cifar100 oxford_iiit_pet imagenet_v2
```

完整的数据集列表请参见[此链接](https://www.tensorflow.org/datasets/catalog/overview#all_datasets)。

某些数据集（如 `imagenet2012` 或 `imagenet2012_real`）需要手动下载数据并放入 `$TFDS_DATA_DIR/downloads/manual/` 目录下（默认为 `~/tensorflow_datasets/downloads/manual/`）。例如，对于 `imagenet2012` 和 `imagenet2012_real`，需要将官方的 `ILSVRC2012_img_train.tar` 和 `ILSVRC2012_img_val.tar` 文件放入该目录，然后运行：
`python3 -m big_vision.tools.download_tfds_datasets imagenet2012 imagenet2012_real`
（可能需要约 1 小时）。

如果您使用 `Google Cloud`（特别是 TPU），则可以将预处理好的数据（存储在 `$TFDS_DATA_DIR` 中）上传到 "Google Cloud Bucket"，并在您的任何（TPU）虚拟机上使用该 bucket 访问数据。

## 在 GPU 机器上运行

最后，在安装完所有 Python 依赖项并准备好 `tfds` 数据后，用户可以使用所选的配置运行作业。例如，要在 ImageNet 数据上训练 `ViT-S/16` 模型，应运行以下命令：

```
python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir workdirs/`date '+%m-%d_%H%M'`
```

或者训练 MLP-Mixer-B/16（请注意 `gpu8` 配置参数，它减少了默认批次大小和轮数）：

```
python3 -m big_vision.train --config big_vision/configs/mlp_mixer_i1k.py:gpu8 --workdir workdirs/`date '+%m-%d_%H%M'`
```

# Cloud TPU VM 配置

## 创建 TPU VM

如需创建一台配备 8 个 TPU 核心的单机，请按照以下 Cloud TPU JAX 文档操作：
https://cloud.google.com/tpu/docs/run-calculation-jax

为了支持大规模视觉研究，建议使用多主机、更多核心的配置。下面我们提供相关操作指南。

首先，创建一些常用变量以便后续复用：

```
export NAME=<TPU 部署名称，例如 my-tpu-machine>
export ZONE=<GCP 地理区域，例如 europe-west4-a>
export GS_BUCKET_NAME=<存储桶名称，例如 my_bucket>
```

以下命令行将创建配备 32 个核心、4 台主机的 TPU VM：

```
gcloud compute tpus tpu-vm create $NAME --zone $ZONE --accelerator-type v3-32 --version tpu-ubuntu2204-base
```

## 在 TPU VM 上安装 `big_vision`

获取 `big_vision` 仓库，将其复制到所有 TPU VM 主机并安装依赖项：

```
git clone https://github.com/google-research/big_vision
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --zone=$ZONE --worker=all
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## 下载并准备 TFDS 数据集

我们建议按上述方式在本地准备 `tfds` 数据，然后将数据上传到 `Google Cloud` 存储桶。不过，如果您愿意，也可以使用 TPU 机器自动准备不需要手动下载的数据集，如下所示。请注意，TPU 机器只有 100 GB 的磁盘空间，且多主机 TPU 切片不允许以写入模式挂载外部磁盘，因此以下说明可能不适用于准备大型数据集。作为另一种替代方案，我们提供了[如何在纯 CPU GCP 机器上准备 `tfds` 数据的说明](#在独立的-gcp-cpu-机器上准备-tfds-数据)。

具体而言，评估期间使用的七个 TFDS 数据集可以通过此命令在 TPU 机器上的 `~/tensorflow_datasets` 下生成：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "TFDS_DATA_DIR=~/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets cifar10 cifar100 oxford_iiit_pet oxford_flowers102 cars196 dtd uc_merced"
```

随后可以将数据集复制到 GS 存储桶，使所有 TPU 工作节点都能访问它们：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "rm -r ~/tensorflow_datasets/downloads && gsutil cp -r ~/tensorflow_datasets gs://$GS_BUCKET_NAME"
```

如果您希望集成其他公开或自定义数据集（如 `imagenet2012`），请遵循[官方指南](https://www.tensorflow.org/datasets/catalog/overview)。

## 预训练模型

有关预训练模型的完整列表，请查看与模型代码定义在同一模块中的 `load` 函数。关于如何使用这些模型的示例配置，请参见 `configs/transfer.py`。

## 在 TPU VM 上运行迁移脚本

以下命令行在 `cifar10` 数据集上微调预训练的 `vit-i21k-augreg-b/32` 模型：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## 在 TPU VM 上运行训练脚本

要在大型数据集（例如 `imagenet2012`，[准备 TFDS 数据集](https://www.tensorflow.org/datasets/catalog/imagenet2012)）上训练您自己的 big_vision 模型，请运行以下命令行：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

## FSDP 训练

`big_vision` 支持灵活的参数与模型分片策略。
目前，我们通过简单的配置修改即可支持主流的 FSDP（Fully Sharded Data Parallel）分片，参见[此配置示例](big_vision/configs/transfer.py)。
例如，要对预训练的 ViT-L 模型运行 FSDP 微调，请运行以下命令（可根据硬件调整批次大小）：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-l/16,dataset=oxford_iiit_pet,crop=resmall_crop,fsdp=True,batch_size=256 --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## 使用 SigLIP 进行图像-文本训练

使用公开 `coco` 标注数据的最小示例：

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_lit_coco.py --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`"
```

## 有时很有用的 gcloud 命令

- 销毁 TPU 机器：`gcloud compute tpus tpu-vm delete $NAME --zone $ZONE`
- 删除所有主机上的所有 big_vision 相关文件夹：`gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'rm -rf ~/big_vision ~/bv_venv'`

## 在独立的 GCP CPU 机器上准备 `tfds` 数据

首先创建一台新机器和一个磁盘（可根据需要调整具体的机器类型和磁盘设置/容量）：

```
export NAME_CPU_HOST=<纯 CPU 机器名称>
export NAME_DISK=<磁盘名称>
gcloud compute instances create $NAME_CPU_HOST --machine-type c3-standard-22 --zone $ZONE --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud
gcloud compute disks create $NAME_DISK --size 1000GB --zone $ZONE --type pd-balanced
```

现在将磁盘挂载到新创建的机器上：

```
gcloud compute instances attach-disk $NAME_CPU_HOST --disk $NAME_DISK --zone $ZONE
```

接下来，SSH 连接到该机器 `gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE` 并[按照说明格式化并挂载磁盘](https://cloud.google.com/compute/docs/disks/format-mount-disk-linux)。假设将其挂载到 `/mnt/disks/tfds`。

差不多完成了，现在克隆并设置 `big_vision`：

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "git clone https://github.com/google-research/big_vision.git && cd big_vision && sh big_vision/run_tpu.sh"
```

最后，使用实用脚本准备数据集（例如 `coco_captions`）并将结果复制到 Google Cloud 存储桶：

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "cd big_vision && TFDS_DATA_DIR=/mnt/disks/tfds/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets coco_captions"
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "rm -rf /mnt/disks/tfds/tensorflow_datasets/downloads && gsutil cp -r /mnt/disks/tfds/tensorflow_datasets gs://$GS_BUCKET_NAME"
```

# ViT 基线

我们在名为 `vit_s16_i1k.py` 的配置文件中提供了一个精心调优的 ViT-S/16 基线。它在 ImageNet 验证集上经过 90 个轮次训练达到了 76.5% 的准确率，是开展 ViT 模型研究的一个强大且简洁的起点。

更多详情请参阅我们的 [arXiv 论文简记](https://arxiv.org/abs/2205.01580)。如果该基线对您的研究有所帮助，请考虑引用：

```
@article{vit_baseline,
  url = {https://arxiv.org/abs/2205.01580},
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Better plain ViT baselines for ImageNet-1k},
  journal={arXiv preprint arXiv:2205.01580},
  year = {2022},
}
```

# 各项目特定 Commit 记录

特定项目代码已知可正常运行的最后提交。核心代码与配置预期可在最新的 HEAD 分支上正常工作。

| 项目 | Commit |
|---|---|
| UViM | https://github.com/google-research/big_vision/commit/21bd6ebe253f070f584d8b777ad76f4abce51bef |
| image_text | https://github.com/google-research/big_vision/commit/8921d5141504390a8a4f7b2dacb3b3c042237290 |
| distill | https://github.com/google-research/big_vision/commit/2f3f493af048dbfd97555ff6060f31a0e686f17f |
| GSAM | WIP |
| CLIPPO | https://github.com/google-research/big_vision/commit/fd2d3bd2efc9d89ea959f16cd2f58ae8a495cd44 |
| CapPa | https://github.com/google-research/big_vision/commit/7ace659452dee4b68547575352c022a2eef587a5 |
| GIVT | https://github.com/google-research/big_vision/commit/0cb70881dd33b3343b769347dc19793c4994b8cb |

# 引用本代码库

如果您发现此代码库对您的研究有所帮助，请考虑使用以下 BibTEX 进行引用：

```
@misc{big_vision,
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Big Vision},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/google-research/big_vision}}
}
```

# 免责声明

这并非官方支持的 Google 产品。

# 开源许可证

除非另有明确说明，big_vision 代码库中的所有内容（包括模型与 Colab）均在 Apache 2.0 许可证下发布。
有关完整许可证文本，请参见 LICENSE 文件。

---

> **免责声明**：本文档由社区志愿者进行本地化翻译，旨在促进跨语言的技术交流与学习。如翻译内容与[英文原版](README.md)存在出入，请以英文原版为准。
