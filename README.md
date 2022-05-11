# Big Vision

This codebase is designed for training large-scale vision models on
[Cloud TPU VMs](https://cloud.google.com/blog/products/compute/introducing-cloud-tpu-vms).
It is based on [Jax](https://github.com/google/jax)/[Flax](https://github.com/google/flax)
libraries, and uses [tf.data](https://www.tensorflow.org/guide/data) and
[TensorFlow Datasets](https://www.tensorflow.org/datasets) for scalable input
pipelines in the Cloud.

The open-sourcing of this codebase has two main purposes:
1. Publishing the code of research projects developed in this codebase (see a
   list below).
2. Providing a strong starting point for running large-scale vision experiments
   on Google Cloud TPUs, which should scale seamlessly and out-of-the box from a
   single TPU core to a distributed setup with up to 2048 TPU cores.

Note, that despite being TPU-centric, our codebase should in general support
CPU, GPU and single-host multi-GPU training, thanks to JAX' well-executed and
transparent support for multiple backends.

`big_vision` aims to support research projects at Google. We are unlikely to
work on feature requests or accept external contributions, unless they were
pre-approved (ask in an issue first). For a well-supported transfer-only
codebase, see also [vision_transformer](https://github.com/google-research/vision_transformer).

The following research projects were originally conducted in the `big_vision`
codebase:

### Architecture research

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929), by
  Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*,
  Xiaohua Zhai*, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
  Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby*
- [Scaling Vision Transformers](https://arxiv.org/abs/2106.04560), by
  Xiaohua Zhai*, Alexander Kolesnikov*, Neil Houlsby, and Lucas Beyer*
- [How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers](https://arxiv.org/abs/2106.10270), by
  Andreas Steiner*, Alexander Kolesnikov*, Xiaohua Zhai*, Ross Wightman,
  Jakob Uszkoreit, and Lucas Beyer*
- [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601), by
  Ilya Tolstikhin*, Neil Houlsby*, Alexander Kolesnikov*, Lucas Beyer*,
  Xiaohua Zhai, Thomas Unterthiner, Jessica Yung, Andreas Steiner,
  Daniel Keysers, Jakob Uszkoreit, Mario Lucic, Alexey Dosovitskiy

### Multimodal research
- [LiT: Zero-Shot Transfer with Locked-image Text Tuning](https://arxiv.org/abs/2111.07991), by
  Xiaohua Zhai*, Xiao Wang*, Basil Mustafa*, Andreas Steiner*, Daniel Keysers,
  Alexander Kolesnikov, and Lucas Beyer*

### Knowledge distillation
- [Knowledge distillation: A good teacher is patient and consistent](https://arxiv.org/abs/2106.05237), by
  Lucas Beyer*, Xiaohua Zhai*, Amélie Royer*, Larisa Markeeva*, Rohan Anil,
  and Alexander Kolesnikov*

### Misc
- [Are we done with ImageNet?](https://arxiv.org/abs/2006.07159), by
  Lucas Beyer*, Olivier J. Hénaff*, Alexander Kolesnikov*, Xiaohua Zhai*,
  and Aäron van den Oord*

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
`configs/` directory. Custom trainers and modules can seamlessly extend/modify
the configuration options.

Training jobs are robust to interruptions and will resume seamlessly from the
last saved checkpoint (assuming user provides the correct `--workdir` path).

Each configuration file contains a comment at the top with a `COMMAND` snippet
to run it, and some hint of expected runtime and results. See below for more
details, but generally speaking, running on a GPU machine involves calling
`python -m COMMAND` while running on TPUs, including multi-host, involves

```
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all
  --command "bash big_vision/run_tpu.sh COMMAND"
```

See instructions below for more details on how to use Google Cloud TPUs.

# Current and future contents

The first release contains the core part of pre-training, transferring, and
evaluating classification models at scale on Cloud TPU VMs.

Features and projects we plan to release in the near future, in no particular
order:
- ImageNet-21k in TFDS.
- MLP-Mixer.
- Loading misc public models used in our publications (NFNet, MoCov3, DINO).
- Contrastive Image-Text model training and evaluation as in LiT and CLIP.
- "Patient and consistent" distillation.
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

# Running on Cloud TPU VMs

## Create TPU VMs

To create a single machine with 8 TPU cores, follow the following Cloud TPU JAX
document:
https://cloud.google.com/tpu/docs/run-calculation-jax

To support large-scale vision research, more cores with multiple hosts are
recommended. Below we provide instructions on how to do it.

First, create some useful variables, which we be reused:

```
export NAME="a name of the TPU deployment, e.g. my-tpu-machine"
export ZONE="GCP geographical zone, e.g. europe-west4-a"
export GS_BUCKET_NAME="Name of the storage bucket, e.g. my_bucket"
```

The following command line will create TPU VMs with 32 cores,
4 hosts.

```
gcloud alpha compute tpus tpu-vm create $NAME --zone $ZONE --accelerator-type v3-32 --version tpu-vm-tf-2.8.0
```

## Install `big_vision` on TPU VMs

Fetch the `big_vision` repository, copy it to all TPU VM hosts, and install
dependencies.

```
git clone --branch=master https://github.com/google-research/big_vision
gcloud alpha compute tpus tpu-vm scp --recurse big_vision/big_vision $NAME: --worker=all --zone=$ZONE
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "bash big_vision/run_tpu.sh"
```

## Download and prepare TFDS datasets

Everything in this section you need to do only once, and, alternatively, you can
also do it on your local machine and copy the result to the cloud bucket. For
convenience, we provide instructions on how to prepare data using Cloud TPUs.

Download and prepare TFDS datasets using a single worker. Seven TFDS datasets
used during evaluations will be generated under `~/tensorflow_datasets/` (should
take 10-15 minutes in total).

```
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "bash big_vision/run_tpu.sh big_vision.tools.download_tfds_datasets cifar10 cifar100 oxford_iiit_pet oxford_flowers102 cars196 dtd uc_merced"
```

Copy the datasets to GS bucket, to make them accessible to all TPU workers.

```
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=0 --command "rm -r ~/tensorflow_datasets/downloads && gsutil cp -r ~/tensorflow_datasets gs://$GS_BUCKET_NAME"
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
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/transfer.py:model=vit-i21k-augreg-b/32,dataset=cifar10,crop=resmall_crop --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'` --config.lr=0.03"
```

## Run the train script on TPU VMs

To train your own big_vision models on a large dataset,
e.g. `imagenet2012` ([prepare the TFDS dataset](https://www.tensorflow.org/datasets/catalog/imagenet2012)),
run the following command line.

```
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
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
