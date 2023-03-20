## Image-and-Language Understanding from Pixels Only

*by Michael Tschannen, Basil Mustafa, Neil Houlsby* [[arxiv]](https://arxiv.org/abs/2212.08045) [[colab]](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb)

We provide pretrained CLIP with Pixels Only (CLIPPO) models and code to train such models on image/alt-text data sets.

### Pretrained models

Six ViT-B/16 models trained on a mix of [`YFCC-100M`](https://arxiv.org/abs/1503.01817) and [`C4`](https://arxiv.org/abs/1910.10683) (some initialized with an [ImageNet21k-pretrained checkpoint](https://github.com/google-research/vision_transformer#vision-transformer)\) are available.
These models were trained using the schedules and hyperparameters described in the paper. We use the full `YFCC-100M` data set, sampling one of the available `title/description/tag` annotations at random for each each example. We drop non-descriptive annotations (e.g. descriptions consisting of digits only) following the filtering procedure outlined in the [LiT paper](https://arxiv.org/abs/2303.04671), Appendix E. The preprocessing for the `C4` data is as described in the paper.

The tables below show details about the checkpoints and their performance on Vision & Language benchmarks, and [`GLUE`](https://arxiv.org/abs/1804.07461). We also provide a [colab](https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/clippo/clippo_colab.ipynb) to load the models, compute embeddings, and perform zero-shot classification.

##### Checkpoint details

| model            | training dataset   | #param.   | steps   | checkpoint |
|:-----------------|:-------------------|:----------|:--------|:-----------|
| CLIPPO           | YFCC-100M          | 93M       | 250k    | `gs://big_vision/clippo/clippo_b16_yfcc100m.npz` |
| CLIPPO I21k init | YFCC-100M          | 93M       | 250k    | `gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init.npz` |
| CLIPPO I21k init | YFCC-100M + 25%C4  | 93M       | 333k    | `gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_25c4.npz` |
| CLIPPO I21k init | YFCC-100M + 50%C4  | 93M       | 500k    | `gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_50c4.npz` |
| CLIPPO I21k init | YFCC-100M + 75%C4  | 93M       | 500k    | `gs://big_vision/clippo/clippo_b16_yfcc100m_i21k_init_75c4.npz` |
| CLIPPO           | C4          | 93M       | 250k    | `gs://big_vision/clippo/clippo_b16_100c4.npz` |

##### Vision \& Language results

| model            | training dataset   | ImageNet 10-shot | ImageNet 0-shot | MS-COCO I→T | MS-COCO T→I |
|:-----------------|:-------------------|-----------:|----------:|--------:|--------:|
| CLIPPO           | YFCC-100M          |       38.2 |      43.4 |    34.7 |    19.7 |
| CLIPPO I21k init | YFCC-100M          |       44.7 |      47.4 |    36.1 |    21.3 |
| CLIPPO I21k init | YFCC-100M + 25%C4  |       43.8 |      44.8 |    33.3 |    19.4 |
| CLIPPO I21k init | YFCC-100M + 50%C4  |       41.2 |      42.0 |    31.4 |    17.8 |
| CLIPPO I21k init | YFCC-100M + 75%C4  |       34.5 |      33.4 |    26.6 |    14.6 |

##### GLUE results

| model            | training dataset   | MNLI-M/MM   |   QQP |   QNLI |   SST-2 |   COLA |   STS-B |   MRPC |   RTE |   avg |
|:-----------------|:-------------------|:------------|------:|-------:|--------:|-------:|--------:|-------:|------:|------:|
| CLIPPO           | YFCC-100M          | 71.3 / 71.5 |  79.1 |   67.9 |    85.7 |    0.0 |    14.0 |   83.4 |  54.9 |  58.6 |
| CLIPPO I21k init | YFCC-100M          | 70.0 / 70.1 |  83.7 |   81.6 |    86.1 |    0.0 |    18.5 |   83.0 |  53.1 |  60.7 |
| CLIPPO I21k init | YFCC-100M + 25%C4  | 75.7 / 75.1 |  85.2 |   83.5 |    89.6 |    0.0 |    82.3 |   82.7 |  52.7 |  69.7 |
| CLIPPO I21k init | YFCC-100M + 50%C4  | 77.4 / 77.4 |  86.0 |   83.9 |    91.7 |   34.5 |    84.5 |   85.1 |  56.3 |  75.2 |
| CLIPPO I21k init | YFCC-100M + 75%C4  | 79.8 / 79.1 |  86.5 |   84.3 |    92.0 |   44.5 |    85.3 |   88.2 |  58.5 |  77.6 |
| CLIPPO           | C4                 | 79.9 / 80.2 |  86.7 |   85.2 |    93.3 |   50.9 |    84.7 |   86.3 |  58.5 |  78.4 |

### Training your own models

To train your own CLIPPO model, please follow the setup instructions in the [`big_vision` main README](https://github.com/google-research/big_vision#cloud-tpu-vm-setup). In the following, we provide the CLIPPO-specific commands required in addition to the setup, assume you are using the Google Cloud TPU setup (potentially with adapted TPU configuration, see table below). If you are using GPUs, please set up your machine directly and only execute the `--command` portions of the commands below from the `big_vision` repository root.

The text rendering preproprocessing function requires manual download of the Unifont .hex files from [Unifoundry](https://unifoundry.com/unifont/) (please follow link for license):

```bash
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all \
--command "bash big_vision/pp/proj/clippo/download_unifont.sh"
```

Launch the training by running

```bash
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all \
--command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.contrastive --config big_vision/configs/proj/clippo/train_clippo.py --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

*Important note:* The input pipeline relies on [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) which does not provide automatic integration with large image/alt-text datasets out of the box. The above config therefore trains by default on MS-COCO Captions which can be automatically downloaded via TFDS, and additionally initializes the CLIPPO ViT backbone with weights pretrained on ImageNet21k. This setup is not meant to produce good accuracy, but to provide the user with a way to sanity-check their setup. If you want to train on a large data set such as [`LAION-400M`](https://arxiv.org/abs/2111.02114) or [`YFCC-100M`](https://arxiv.org/abs/1503.01817), please follow [these instructions](https://www.tensorflow.org/datasets/add_dataset) to wrap your data set using TFDS, and update the dataset in the config accordingly. Also note that the ImageNet1k evaluations require manual download of the data, see [these instructions](https://github.com/google-research/big_vision#preparing-tfds-data). To train with your own data set and with ImageNet1k-based evaluations, use `--config big_vision/configs/proj/clippo/train_clippo.py:test_with_coco=False,i1k_eval=True` in the command above.

##### Expected results

| train dataset | batch size | #steps | TPU chips | ImageNet 0-shot | MS-COCO I→T | MS-COCO T→I | Config `arg` |
| :---  | ---:         | ---: | ---: | :---:           | :---:       | :---:       | :---         |
| *MS-COCO (sanity check)* | 4000 | 400 | 32 v3 | 4.2 | 12.6 | 8.6 | `i1k_eval=True` |
| LAION-400M | 8192 | 100k |128 v2 | 51.5 | 44.8 | 29.3 | `test_with_coco=False,i1k_eval=True` |
| LAION-400M | 10240\* | 100k | 128 v3 | 53.6 | 46.7 | 30.3 | `test_with_coco=False,i1k_eval=True` |

\* The experiments in the paper use a batch size of 10240 which requires a memory-optimized ViT implementation to run on 128 TPU v2 chips or 128 TPU v3 chips (in which case the TPU memory capacity allows to increase the batch size beyond 10240).

### Citation

```
@inproceedings{tschannen2023image,
  title={Image-and-Language Understanding from Pixels Only},
  author={Tschannen, Michael and Mustafa, Basil and Houlsby, Neil},
  booktitle={Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```
