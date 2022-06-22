# Knowledge distillation: A good teacher is patient and consistent
*by Lucas Beyer, Xiaohua Zhai, Amélie Royer, Larisa Markeeva, Rohan Anil, Alexander Kolesnikov*

## Introduction
We publish all teacher models, and configurations for the main experiments of
the paper, as well as training logs and student models.

Please read the main [big_vision README](/README.md) to learn how to run
configs, and remember that each config file contains an example invocation in
the top-level comment.

## Results

We provide the following [colab to read and plot the logfiles](https://colab.research.google.com/drive/1nMykzUzsfQ_uAxfj3k35DYsATnG_knPl?usp=sharing)
of a few runs that we reproduced on Cloud.

### ImageNet-1k

The file [bit_i1k.py](bit_i1k.py) is the configuration which reproduces our
distillation runs on ImageNet-1k reported in Figures 1 and 5(left) and the first
row of Table1.

We release both student and teacher models:

| Model      | Download link | Resolution  | ImageNet top-1 acc. (paper) | 
| :---       | :---:         | :---:       |  :---:                      |
| BiT-R50x1  | [link](https://storage.googleapis.com/bit_models/distill/R50x1_160.npz)      | 160 |  80.5 |
| BiT-R50x1  | [link](https://storage.googleapis.com/bit_models/distill/R50x1_224.npz)      | 224 |  82.8 |
| BiT-R152x2 | [link](https://storage.googleapis.com/bit_models/distill/R152x2_T_224.npz)   | 224 |  83.0 |
| BiT-R152x2 | [link](https://storage.googleapis.com/bit_models/distill/R152x2_T_384.npz)   | 384 |  84.3 |

### Flowers/Pet/Food/Sun

The files [bigsweep_flowers_pet.py](bigsweep_flowers_pet.py) and
[bigsweep_food_sun.py](bigsweep_food_sun.py) can be used to reproduce the
distillation runs on these datasets and shown in Figures 3,4,9-12, and Table4.

While our open-source release does not currently support doing hyper-parameter
sweeps, we still provide an example of the sweeps at the end of the configs
for reference.

### Teacher models
Links to all teacher models we used can be found in [common.py](common.py).
