# Big Vision

This codebase is designed for training large-scale vision models using
[Cloud TPU VMs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms)
or GPU machines. It is based on [Jax](https://github.com/jax-ml/jax)/[Flax](https://github.com/google/flax)
libraries, and uses [tf.data](https://www.tensorflow.org/guide/data) and
[TensorFlow Datasets](https://www.tensorflow.org/datasets) for scalable and
reproducible input pipelines.

The open-sourcing of this codebase has two main purposes:
1. Publishing the code of research projects developed in this codebase (see a
   list below).
2. Providing a strong starting point for running large-scale vision experiments
   on GPU machines and Google Cloud TPUs, which should scale seamlessly and
   out-of-the box from a single TPU core to a distributed setup with up to 2048
   TPU cores.

`big_vision` aims to support research projects at Google. We are unlikely to
work on feature requests or accept external contributions, unless they were
pre-approved (ask in an issue first). For a well-supported transfer-only
codebase, see also [vision_transformer](https://github.com/google-research/vision_transformer).

Note that `big_vision` is quite dynamic codebase and, while we intend to keep
the core code fully-functional at all times, we can not guarantee timely updates
of the project-specific code that lives in the `.../proj/...` subfolders.
However, we provide a [table](#project-specific-commits) with last known
commits where specific projects were known to work.

The following research projects were originally conducted in the `big_vision`
codebase:

### Architecture research

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by
  Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
  Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
  Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby*
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560), by
  Xiaohua Zhai*, Alexander Kolesnikov*, Neil Houlsby, and Lucas Beyer*\
  Resources: [config](big_vision/configs/proj/scaling_laws/train_vit_g.py).
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270), by
  Andreas Steiner*, Alexander Kolesnikov*, Xiaohua Zhai*, Ross Wightman,
  Jakob Uszkoreit, and Lucas Beyer*
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by
  Ilya Tolstikhin*, Neil Houlsby*, Alexander Kolesnikov*, Lucas Beyer*,
  Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
  Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy\
  Resources: [config](big_vision/configs/mlp_mixer_i1k.py).
- [Better plain ViT baselines for ImageNet-1k](https://arxiv.org/abs/2205.01580), by
  Lucas Beyer, Xiaohua Zhai, Alexander Kolesnikov\
  Resources: [config](big_vision/configs/vit_s16_i1k.py)
- [UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes](https://arxiv.org/abs/2205.10337), by
  Alexander Kolesnikov^*, André Susano Pinto^*, Lucas Beyer*, Xiaohua Zhai*, Jeremiah Harmsen*, Neil Houlsby*\
  Resources: [readme](big_vision/configs/proj/uvim/README.md), [configs](big_vision/configs/proj/uvim), [colabs](big_vision/configs/proj/uvim).
- [FlexiViT: One Model for All Patch Sizes](https://arxiv.org/abs/2212.08013), by
  Lucas Beyer*, Pavel Izmailov*, Alexander Kolesnikov*, Mathilde Caron*, Simon
  Kornblith*, Xiaohua Zhai*, Matthias Minderer*, Michael Tschannen*, Ibrahim
  Alabdulmohsin*, Filip Pavetic*\
  Resources: [readme](big_vision/configs/proj/flexivit/README.md), [configs](big_vision/configs/proj/flexivit).
- [Dual PatchNorm](https://arxiv.org/abs/2302.01327), by Manoj Kumar, Mostafa Dehghani, Neil Houlsby.
- [Getting ViT in Shape: Scaling Laws for Compute-Optimal Model Design](https://arxiv.org/abs/2305.13035), by
  Ibrahim Alabdulmohsin*, Xiaohua Zhai*, Alexander Kolesnikov, Lucas Beyer*.
- (partial) [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442), by
  Mostafa Dehghani*, Josip Djolonga*, Basil Mustafa*, Piotr Padlewski*, Jonathan Heek*, *wow many middle authors*, Neil Houlsby*.
- (partial) [Finite Scalar Quantization: VQ-VAE Made Simple](https://arxiv.org/abs/2309.15505), by
  Fabian Mentzer, David Minnen, Eirikur Agustsson, Michael Tschannen.
- [GIVT: Generative Infinite-Vocabulary Transformers](https://arxiv.org/abs/2312.02116), by
  Michael Tschannen, Cian Eastwood, Fabian Mentzer.\
  Resources: [readme](big_vision/configs/proj/givt/README.md), [config](big_vision/configs/proj/givt/givt_imagenet2012.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/givt/givt_demo_colab.ipynb).
- [Unified Auto-Encoding with Masked Diffusion](https://arxiv.org/abs/2406.17688), by
  Philippe Hansen-Estruch, Sriram Vishwanath, Amy Zhang, Manan Tomar.
- [Jet: A Modern Transformer-Based Normalizing Flow](https://arxiv.org/abs/2412.15129), by
  Alexander Kolesnikov*, André Susano Pinto*, Michael Tschannen*, [configs](big_vision/configs/proj/jet)
- [JetFormer: An autoregressive generative model of raw images and text](https://arxiv.org/abs/2411.19722), by
  Michael Tschannen*, André Susano Pinto*, Alexander Kolesnikov*. [configs](big_vision/configs/proj/jetformer).


### Multimodal research

- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991), by
  Xiaohua Zhai*, Xiao Wang*, Basil Mustafa*, Andreas Steiner*, Daniel Keysers,
  Alexander Kolesnikov, and Lucas Beyer*\
  Resources: [trainer](big_vision/trainers/proj/image_text/contrastive.py), [config](big_vision/configs/proj/image_text/lit_coco.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/lit.ipynb).
- [CLIPPO: Image-and-Language Understanding from Pixels Only](https://arxiv.org/abs/2212.08045), by
  Michael Tschannen, Basil Mustafa, Neil Houlsby\
  Resources: [readme](big_vision/configs/proj/clippo/README.md), [config](big_vision/configs/proj/clippo/train_clippo.py), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb).
- [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343), by
  Xiaohua Zhai*, Basil Mustafa, Alexander Kolesnikov, Lucas Beyer*\
  Resources: [colab and models](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP_demo.ipynb), code TODO.
- [A Study of Autoregressive Decoders for Multi-Tasking in Computer Vision](https://arxiv.org/abs/2303.17376), by
  Lucas Beyer*, Bo Wan*, Gagan Madan*, Filip Pavetic*, Andreas Steiner*, Alexander Kolesnikov, André Susano Pinto, Emanuele Bugliarello, Xiao Wang, Qihang Yu, Liang-Chieh Chen, Xiaohua Zhai*.
- [Image Captioners Are Scalable Vision Learners Too](https://arxiv.org/abs/2306.07915), by
  Michael Tschannen*, Manoj Kumar*, Andreas Steiner*, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*.\
  Resources: [readme](big_vision/configs/proj/cappa/README.md), [config](big_vision/configs/proj/cappa/pretrain.py), [model](big_vision/models/proj/cappa/cappa.py).
- [Three Towers: Flexible Contrastive Learning with Pretrained Image Models](https://arxiv.org/abs/2305.16999), by Jannik Kossen, Mark Collier, Basil Mustafa, Xiao Wang, Xiaohua Zhai, Lucas Beyer, Andreas Steiner, Jesse Berent, Rodolphe Jenatton, Efi Kokiopoulou.
- (partial) [PaLI: A Jointly-Scaled Multilingual Language-Image Model](https://arxiv.org/abs/2209.06794), by Xi Chen, Xiao Wang, Soravit Changpinyo, *wow so many middle authors*, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut.
- (partial) [PaLI-3 Vision Language Models: Smaller, Faster, Stronger](https://arxiv.org/abs/2310.09199), by Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut.
- [LocCa](https://arxiv.org/abs/2403.19596), by
  Bo Wan, Michael Tschannen, Yongqin Xian, Filip Pavetic, Ibrahim Alabdulmohsin, Xiao Wang, André Susano Pinto, Andreas Steiner, Lucas Beyer, Xiaohua Zhai.
- [PaliGemma](https://arxiv.org/abs/2407.07726),
  [PaliGemma 2](https://arxiv.org/abs/2412.03555), by *wow many authors*.\
- Resources: [readme](big_vision/configs/proj/paligemma/README.md),
    [model](big_vision/models/proj/paligemma/paligemma.py),
    [transfer configs](big_vision/configs/proj/paligemma/transfers),
    [datasets](big_vision/datasets),
    [CountBenchQA](big_vision/datasets/countbenchqa/data/countbench_paired_questions.json).
- [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786), by *wow many authors*.\
  Resources: [readme (with checkpoints)](big_vision/configs/proj/image_text/README_siglip2.md), [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/SigLIP2_demo.ipynb).

### Training

- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), by
  Lucas Beyer*, Xiaohua Zhai*, Amélie Royer*, Larisa Markeeva*, Rohan Anil,
  and Alexander Kolesnikov*\
  Resources: [README](big_vision/configs/proj/distill/README.md), [trainer](big_vision/trainers/proj/distill/distill.py), [colab](https://colab.research.google.com/drive/1nMykzUzsfQ_uAxfj3k35DYsATnG_knPl?usp=sharing).
- [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412), by
  Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur
- [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/abs/2203.08065), by Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha Dvornek, Sekhar Tatikonda, James Duncan and Ting Liu \
  Resources: [trainer](big_vision/trainers/proj/gsam/gsam.py), [config](big_vision/configs/proj/gsam/vit_i1k_gsam_no_aug.py) [reproduced results](https://github.com/google-research/big_vision/pull/8#pullrequestreview-1078557411)
- [Tuning computer vision models with task rewards](https://arxiv.org/abs/2302.08242), by
  André Susano Pinto*, Alexander Kolesnikov*, Yuge Shi, Lucas Beyer, Xiaohua Zhai.
- (partial) [VeLO: Training Versatile Learned Optimizers by Scaling Up](https://arxiv.org/abs/2211.09760) by
  Luke Metz, James Harrison, C. Daniel Freeman, Amil Merchant, Lucas Beyer, James Bradbury, Naman Agrawal, Ben Poole, Igor Mordatch, Adam Roberts, Jascha Sohl-Dickstein.

### Misc

- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159), by
  Lucas Beyer*, Olivier J. Hénaff*, Alexander Kolesnikov*, Xiaohua Zhai*, Aäron van den Oord*.
- [No Filter: Cultural and Socioeconomic Diversity in Contrastive Vision-Language Models](https://arxiv.org/abs/2405.13777), by
  Angéline Pouget, Lucas Beyer, Emanuele Bugliarello, Xiao Wang, Andreas Peter Steiner, Xiaohua Zhai, Ibrahim Alabdulmohsin.

# Codebase high-level organization and principles in a nutshell

The main entry point is a trainer module, which typically does all the
boilerplate related to creating a model and an optimizer, loading the data,
checkpointing and training/evaluating the model inside a loop. We provide the
canonical trainer `train.py` in the root folder. Normally, individual projects
within `big_vision` fork and customize this trainer.

All models, evaluators and preprocessing operations live in the corresponding
subdirectories and can often be reused between different projects. We encourage
compatible APIs within these directories to facilitate reusability, but it is
not strictly enforced, as individual projects may need to introduce their custom
APIs.

We have a powerful configuration system, with the configs living in the
`configs/` directory. Custom trainers and modules can directly extend/modify
the configuration options.

Project-specific code resides in the `.../proj/...` namespace. It is not always
possible to keep project-specific in sync with the core `big_vision` libraries,
Below we provide the [last known commit](#project-specific-commits)
for each project where the project code is expected to work.

Training jobs are robust to interruptions and will resume seamlessly from the
last saved checkpoint (assuming a user provides the correct `--workdir` path).

Each configuration file contains a comment at the top with a `COMMAND` snippet
to run it, and some hint of expected runtime and results. See below for more
details, but generally speaking, running on a GPU machine involves calling
`python -m COMMAND` while running on TPUs, including multi-host, involves

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all
  --command "bash big_vision/run_tpu.sh COMMAND"
```

See instructions below for more details on how to run `big_vision` code on a
GPU machine or Google Cloud TPU.

By default we write checkpoints and logfiles. The logfiles are a list of JSON
objects, and we provide a short and straightforward [example colab to read
and display the logs and checkpoints](https://colab.research.google.com/drive/1R_lvV542WUp8Q2y8sbyooZOGCplkn7KI?usp=sharing).

# Current and future contents

The first release contains the core part of pre-training, transferring, and
evaluating classification models at scale on Cloud TPU VMs.

We have since added the following key features and projects:
- Contrastive Image-Text model training and evaluation as in LiT and CLIP.
- Patient and consistent distillation.
- Scaling ViT.
- MLP-Mixer.
- UViM.

Features and projects we plan to release in the near future, in no particular
order:
- ImageNet-21k in TFDS.
- Loading misc public models used in our publications (NFNet, MoCov3, DINO).
- Memory-efficient Polyak-averaging implementation.
- Advanced JAX compute and memory profiling. We are using internal tools for
    this, but may eventually add support for the publicly available ones.

We will continue releasing code of our future publications developed within
`big_vision` here.

### Non-content

The following exist in the internal variant of this codebase, and there is no
plan for their release:
- Regular regression tests for both quality and speed. They rely heavily on
    internal infrastructure.
- Advanced logging, monitoring, and plotting of experiments. This also relies
    heavily on internal infrastructure. However, we are open to ideas on this
    and may add some in the future, especially if implemented in a
    self-contained manner.
- Not yet published, ongoing research projects.


# GPU Setup

We first discuss how to setup and run `big_vision` on a (local) GPU machine,
and then discuss the setup for Cloud TPUs. Note that data preparation step for
(local) GPU setup can be largely reused for the Cloud TPU setup. While the
instructions skip this for brevity, we highly recommend using a
[virtual environment](https://docs.python.org/3/library/venv.html) when
installing python dependencies.

## Setting up python packages

The first step is to checkout `big_vision` and install relevant python
dependencies:

```
git clone https://github.com/google-research/big_vision
cd big_vision/
pip3 install --upgrade pip
pip3 install -r big_vision/requirements.txt
```

The latest version of `jax` library can be fetched as

```
pip3 install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

You may need a different `jax` package, depending on CUDA and cuDNN libraries
installed on your machine. Please consult
[official jax documentation](https://github.com/jax-ml/jax#pip-installation-gpu-cuda)
for more information.

## Preparing tfds data

For unified and reproducible access to standard datasets we opted to use the
`tensorflow_datasets` (`tfds`) library. It requires each dataset to be
downloaded, preprocessed and then to be stored on a hard drive (or, if you use
"Google Cloud", preferably stored in a "GCP bucket".).

Many datasets can be downloaded and preprocessed automatically when used
for the first time. Nevertheless, we intentionally disable this feature and
recommend doing dataset preparation step separately, ahead of the first run. It
will make debugging easier if problems arise and some datasets, like
`imagenet2012`, require manually downloaded data.

Most of the datasets, e.g. `cifar100`, `oxford_iiit_pet` or `imagenet_v2`
can be fully automatically downloaded and prepared by running

```
cd big_vision/
python3 -m big_vision.tools.download_tfds_datasets cifar100 oxford_iiit_pet imagenet_v2
```

A full list of datasets is available at [this link](https://www.tensorflow.org/datasets/catalog/overview#all_datasets).

Some datasets, like `imagenet2012` or `imagenet2012_real`, require the data to
be downloaded manually and placed into `$TFDS_DATA_DIR/downloads/manual/`,
which defaults to `~/tensorflow_datasets/downloads/manual/`. For example, for
`imagenet2012` and `imagenet2012_real` one needs to place the official
`ILSVRC2012_img_train.tar` and `ILSVRC2012_img_val.tar` files in that directory
and then run
`python3 -m big_vision.tools.download_tfds_datasets imagenet2012 imagenet2012_real`
(which may take ~1 hour).

If you use `Google Cloud` and, TPUs in particular, you can then upload
the preprocessed data (stored in `$TFDS_DATA_DIR`) to
"Google Cloud Bucket" and use the bucket on any of your (TPU) virtual
machines to access the data.

## Running on a GPU machine

Finally, after installing all python dependencies and preparing `tfds` data,
the user can run the job using config of their choice, e.g. to train `ViT-S/16`
model on ImageNet data, one should run the following command:

```
python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir workdirs/`date '+%m-%d_%H%M'`
```

or to train MLP-Mixer-B/16, run (note the `gpu8` config param that reduces the default batch size and epoch count):

```
python3 -m big_vision.train --config big_vision/configs/mlp_mixer_i1k.py:gpu8 --workdir workdirs/`date '+%m-%d_%H%M'`
```

# Cloud TPU VM setup

## Create TPU VMs

To create a single machine with 8 TPU cores, follow the following Cloud TPU JAX
document:
https://cloud.google.com/tpu/docs/run-calculation-jax

To support large-scale vision research, more cores with multiple hosts are
recommended. Below we provide instructions on how to do it.

First, create some useful variables, which we be reused:

```
export NAME=<a name of the TPU deployment, e.g. my-tpu-machine>
export ZONE=<GCP geographical zone, e.g. europe-west4-a>
export GS_BUCKET_NAME=<Name of the storage bucket, e.g. my_bucket>
```

The following command line will create TPU VMs with 32 cores,
4 hosts.

```
gcloud compute tpus tpu-vm create $NAME --zone $ZONE --accelerator-type v3-32 --version tpu-ubuntu2204-base
```

## Install `big_vision` on TPU VMs

Fetch the `big_vision` repository, copy it to all TPU VM hosts, and install
dependencies.

```
git clone https://github.com/google-research/big_vision
gcloud compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --zone=$ZONE --worker=all
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## Download and prepare TFDS datasets

We recommend preparing `tfds` data locally as described above and then uploading
the data to `Google Cloud` bucket. However, if you prefer, the datasets which
do not require manual downloads can be prepared automatically using a TPU
machine as described below. Note that TPU machines have only 100 GB of disk
space, and multihost TPU slices do not allow for external disks to be attached
in a write mode, so the instructions below may not work for preparing large
datasets. As yet another alternative, we provide instructions
[on how to prepare `tfds` data on CPU-only GCP machine](#preparing-tfds-data-on-a-standalone-gcp-cpu-machine).

Specifically, the seven TFDS datasets used during evaluations will be generated
under `~/tensorflow_datasets` on TPU machine with this command:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "TFDS_DATA_DIR=~/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets cifar10 cifar100 oxford_iiit_pet oxford_flowers102 cars196 dtd uc_merced"
```

You can then copy the datasets to GS bucket, to make them accessible to all TPU workers.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "rm -r ~/tensorflow_datasets/downloads && gsutil cp -r ~/tensorflow_datasets gs://$GS_BUCKET_NAME"
```

If you want to integrate other public or custom datasets, i.e. imagenet2012,
please follow [the official guideline](https://www.tensorflow.org/datasets/catalog/overview).

## Pre-trained models

For the full list of pre-trained models check out the `load` function defined in
the same module as the model code. And for example config on how to use these
models, see `configs/transfer.py`.

## Run the transfer script on TPU VMs

The following command line fine-tunes a pre-trained `vit-i21k-augreg-b/32` model
on `cifar10` dataset.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Run the train script on TPU VMs

To train your own big_vision models on a large dataset,
e.g. `imagenet2012` ([prepare the TFDS dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012)),
run the following command line.

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

## FSDP training.

`big_vision` supports flexible parameter and model sharding strategies.
Currently, we support a popular FSDP sharding via a simple config change, see [this config example](big_vision/configs/transfer.py).
For example, to run FSDP finetuning of a pretrained ViT-L model, run the following command (possible adjusting batch size depending on your hardware):

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-l/16,dataset=oxford_iiit_pet,crop=resmall_crop,fsdp=True,batch_size=256 --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Image-text training with SigLIP.

A minimal example that uses public `coco` captions data:

```
gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.siglip --config big_vision/configs/proj/image_text/siglip_lit_coco.py --workdir gs://$GS_BUCKET_NAME/big_vision/`date '+%Y-%m-%d_%H%M'`"
```



## Sometimes useful gcloud commands

- Destroy the TPU machines: `gcloud compute tpus tpu-vm delete $NAME --zone $ZONE`
- Remove all big_vision-related folders on all hosts: `gcloud compute tpus tpu-vm ssh $NAME --zone $ZONE --worker=all --command 'rm -rf ~/big_vision ~/bv_venv'`

## Preparing `tfds` data on a standalone GCP CPU machine.

First create a new machine and a disk (feel free to adjust exact machine type and disk settings/capacity):

```
export NAME_CPU_HOST=<A name of a CPU-only machine>
export NAME_DISK=<A name of a disk>
gcloud compute instances create $NAME_CPU_HOST --machine-type c3-standard-22 --zone $ZONE --image-family ubuntu-2204-lts --image-project ubuntu-os-cloud
gcloud compute disks create $NAME_DISK --size 1000GB --zone $ZONE --type pd-balanced
```

Now attach the disk to the newly create machine:

```
gcloud compute instances attach-disk $NAME_CPU_HOST --disk $NAME_DISK --zone $ZONE
```

Next, `ssh` to the machine `gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE` and
[follow instructions to format and mount the disk](https://cloud.google.com/compute/docs/disks/format-mount-disk-linux).
Let's assume it was mounted to `/mnt/disks/tfds`.

Almost there, now clone and set up `big_vision`:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "git clone https://github.com/google-research/big_vision.git && cd big_vision && sh big_vision/run_tpu.sh"
```

Finally, prepare the dataset (e.g. `coco_captions`) using the utility script and
copy the result to you google cloud bucket:

```
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "cd big_vision && TFDS_DATA_DIR=/mnt/disks/tfds/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets coco_captions"
gcloud compute ssh $NAME_CPU_HOST --zone=$ZONE --command "rm -rf /mnt/disks/tfds/tensorflow_datasets/downloads && gsutil cp -r /mnt/disks/tfds/tensorflow_datasets gs://$GS_BUCKET_NAME"
```


# ViT baseline

We provide a well-tuned ViT-S/16 baseline in the config file named
`vit_s16_i1k.py`. It achieves 76.5% accuracy on ImageNet validation split in
90 epochs of training, being a strong and simple starting point for research
on the ViT models.

Please see our [arXiv note](https://arxiv.org/abs/2205.01580) for more details
and if this baseline happens to by useful for your research, consider citing

```
@article{vit_baseline,
  url = {https://arxiv.org/abs/2205.01580},
  author = {Beyer, Lucas and Zhai, Xiaohua and Kolesnikov, Alexander},
  title = {Better plain ViT baselines for ImageNet-1k},
  journal={arXiv preprint arXiv:2205.01580},
  year = {2022},
}
```

# Project specific commits

The last known commit where the specific project code is expected to work. The
core code and configs are expected to work at head.

| Project    | Commit                                                                                        |
|------------|-----------------------------------------------------------------------------------------------|
| UViM       | https://github.com/google-research/big_vision/commit/21bd6ebe253f070f584d8b777ad76f4abce51bef |
| image_text | https://github.com/google-research/big_vision/commit/8921d5141504390a8a4f7b2dacb3b3c042237290 |
| distill    | https://github.com/google-research/big_vision/commit/2f3f493af048dbfd97555ff6060f31a0e686f17f |
| GSAM       | WIP                                                                                           |
| CLIPPO     | https://github.com/google-research/big_vision/commit/fd2d3bd2efc9d89ea959f16cd2f58ae8a495cd44 |
| CapPa      | https://github.com/google-research/big_vision/commit/7ace659452dee4b68547575352c022a2eef587a5 |
| GIVT       | https://github.com/google-research/big_vision/commit/0cb70881dd33b3343b769347dc19793c4994b8cb |

# Citing the codebase

If you found this codebase useful for your research, please consider using
the following BibTEX to cite it:

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

# Disclaimer

This is not an official Google Product.

# License

Unless explicitly noted otherwise, everything in the big_vision codebase
(including models and colabs) is released under the Apache2 license.
See the LICENSE file for the full license text.
