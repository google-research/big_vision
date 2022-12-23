## Image-and-Language Understanding from Pixels Only

*by Michael Tschannen, Basil Mustafa, Neil Houlsby* [[arxiv]](https://arxiv.org/abs/2212.08045)

We provide code to train CLIP with Pixels Only (CLIPPO) models on image/alt-text data sets.

To train your own CLIPPO model, please follow the setup instructions in the [`big_vision` main README](https://github.com/google-research/big_vision#cloud-tpu-vm-setup). In the following, we provide the CLIPPO-specific commands required in addition to the setup, assume you are using the Google Cloud TPU setup (potentially with adapted TPU configuration, see table below). If you are using GPUs, please set up your machine directly and only execute the `--command` portions of the commands below from the `big_vision` repository root.

The text rendering preproprocessing function requires manual download of the Unifont .hex files from [Unifoundry](https://unifoundry.com/unifont/) (please follow link for license).:

```bash
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all \
--command "bash big_vision/pp/proj/clippo/download_unifont.sh"
```

Launch the training by running

```bash
gcloud alpha compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all \
--command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.trainers.proj.image_text.contrastive --config big_vision/configs/proj/clippo/train_clippo.py --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"
```

*Important note:* The input pipeline relies on [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets) which does not provide automatic integration with large image/alt-text datasets out of the box. The above config therefore trains by default on MS-COCO Captions which can be automatically downloaded via TFDS, and additionally initializes the CLIPPO ViT backbone with weights pretrained on ImageNet21k. This setup is not meant to produce good accuracy, but to provide the user with a way to sanity-check their setup. If you want to train on a large data set such as [`LAION-400M`](https://arxiv.org/abs/2111.02114) or [`YFCC100M`](https://arxiv.org/abs/1503.01817), please follow [these instructions](https://www.tensorflow.org/datasets/add_dataset) to wrap your data set using TFDS, and update the dataset in the config accordingly. Also note that the ImageNet1k evaluations require manual download of the data, see [these instructions](https://github.com/google-research/big_vision#preparing-tfds-data). To train with your own data set and with ImageNet1k-based evaluations, use `--config big_vision/configs/proj/clippo/train_clippo.py:test_with_coco=False,i1k_eval=True` in the command above.

#### Expected results

| train dataset | batch size | #steps | TPU chips | ImageNet 0-shot | MS-COCO I→T | MS-COCO T→I | Config `arg` |
| :---  | ---:         | ---: | ---: | :---:           | :---:       | :---:       | :---         |
| *MS-COCO (sanity check)* | 4000 | 400 | 32 v3 | 4.2 | 12.6 | 8.6 | `i1k_eval=True` |
| LAION-400M | 8192 | 100k |128 v2 | 51.5 | 44.8 | 29.3 | `test_with_coco=False,i1k_eval=True` |
| LAION-400M | 10240\* | 100k | 128 v3 | 53.6 | 46.7 | 30.3 | `test_with_coco=False,i1k_eval=True` |

\* The experiments in the paper use a batch size of 10240 which requires a memory-optimized ViT implementation to run on 128 TPU v2 chips or 128 TPU v3 chips (in which case the TPU memory capacity allows to increase the batch size beyond 10240).

#### Citation

```
@article{tschannen2022image,
  title={Image-and-Language Understanding from Pixels Only},
  author={Tschannen, Michael and Mustafa, Basil and Houlsby, Neil},
  journal={arXiv preprint arXiv:2212.08045},
  year={2022}
}
```
