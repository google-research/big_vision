# UViM: A Unified Modeling Approach for Vision with Learned Guiding Codes

*by Alexander Kolesnikov, André Susano Pinto, Lucas Beyer, Xiaohua Zhai, Jeremiah Harmsen, Neil Houlsby*

We provide pretrained UViM models from the [original paper](https://arxiv.org/abs/2205.10337),
as well as the instructions on how to reproduce core paper experiments.

## Pretrained models

The table below contains UViM models (stage I and II) trained for three
different tasks: panoptic segmentation, colorization and depth prediction.

| task                  | model               | dataset                                                                  | accuracy     | download link                                                                             |
| --------------------- | ------------------- | ------------------------------------------------------------------------ | ------------ | ----------------------------------------------------------------------------------------- |
| Panoptic segmentation | UViM Stage I model  | [COCO(2017)](https://cocodataset.org/#home)                              |  75.8 PQ     | [link](https://storage.googleapis.com/big_vision/uvim/panoptic_stageI_params.npz)         |
| Panoptic segmentation | UViM Stage II model | [COCO(2017)](https://cocodataset.org/#home)                              |  43.1 PQ     | [link](https://storage.googleapis.com/big_vision/uvim/panoptic_stageII_params.npz)        |
| Colorization          | UViM Stage I model  | [ILSVRC-2012](https://www.image-net.org/)                                |  15.59 FID   | [link](https://storage.googleapis.com/big_vision/uvim/color_stageI_params.npz)            |
| Colorization          | UViM Stage II model | [ILSVRC-2012](https://www.image-net.org/)                                |  16.99 FID   | [link](https://storage.googleapis.com/big_vision/uvim/color_stageII_params.npz)           |
| Depth                 | UViM Stage I model  | [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |  0.155 RMSE  | [link](https://storage.googleapis.com/big_vision/uvim/depth_stageI_params.npz)            |
| Depth                 | UViM Stage II model | [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) |  0.463 RMSE  | [link](https://storage.googleapis.com/big_vision/uvim/depth_stageII_params.npz)           |

All of this models can be interactively explored in our [colabs](configs/proj/uvim).

## Running on a single-host TPU machine

Below we provide instructions on how to run UViM training (stage I and
stage II) using a single TPU host with 8 TPU accelerators. These instructions
can be easily adapted to a GPU host and multi-host TPU setup, see the main
`big_vision` [README file](README.md).

We assume that the user has already created and `ssh`-ed to the TPU host
machine. The next step is to clone `big_vision` repository:
`git clone https://github.com/google-research/big_vision.git`.

The next steps are to create a python virtual environment and install python
dependencies:
```
virtualenv bv
source bv/bin/activate
cd big_vision/
pip3 install --upgrade pip
pip3 install -r big_vision/requirements.txt
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

After this invoke the helper tool to download and prepare data:
`python3 -m big_vision.tools.download_tfds_datasets coco/2017_panoptic nyu_depth_v2`.
For preparing the ImageNet dataset consult the main codebase README.

> :warning: TPU machines have 100 GB of the disk space. It may not be enough to
> store all training data (though only panoptic or only depth data may fit).
> Consider preparing the data on a seperate machine and then copying it to
> to TPU machine's extra persistent disk or to a Google Cloud Bucket. See
> instructions for [creating an extra persistent disk](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).
> Remember to set the correct data home directory, e.g.`export DISK=/mnt/disk/persist; export TFDS_DATA_DIR=$DISK/tensorflow_datasets`.

Our panoptic evaluator uses raw variant of the COCO data, so we move it into a
separate folder. Note, `tfds` has already pre-downloaded the panoptic data,
except for one small json file that we fetch manually:
```
mkdir $DISK/coco_data
cd $DISK/coco_data
mv $TFDS_DATA_DIR/downloads/extracted/ZIP.image.cocod.org_annot_panop_annot_train<REPLACE_ME_WITH_THE_HASH_CODE>.zip/annotations/* .
wget https://raw.githubusercontent.com/cocodataset/panopticapi/master/panoptic_coco_categories.json
export COCO_DATA_DIR=$DISK/coco_data
```

For FID evaluator, which is used for the colorization model, set the path to the
directory with image id files, e.g.
`export FID_DATA_DIR=<ROOT>/big_vision/evaluators/proj/uvim/coltran_fid_data`.

As an example, stage I panoptic training can be invoked as (note the `:singlehost` config parameter which will use lightweight configuration suitable for a single host):
```
python3 -m big_vision.trainers.proj.uvim.vqvae --config big_vision/configs/proj/uvim/vqvae_coco_panoptic.py:singlehost --workdir workdirs/`date '+%m-%d_%H%M'`
```
or stage II training
```
python3 -m big_vision.trainers.proj.uvim.train --config big_vision/configs/proj/uvim/train_coco_panoptic_pretrained.py:singlehost --workdir workdirs/`date '+%m-%d_%H%M'`
```

## Acknowledgments
The sampling code in `models/proj/uvim/decode.py` module is based on contributions
from Anselm Levskaya, Ilya Tolstikhin and Maxim Neumann.

