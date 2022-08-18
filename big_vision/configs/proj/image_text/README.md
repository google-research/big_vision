# Image/text models

## LiT: Zero-Shot Transfer with Locked-image text Tuning

*by Xiaohua Zhai, Xiao Wang, Basil Mustafa, Andreas Steiner, Daniel Keysers, Alexander Kolesnikov, Lucas Beyer*

https://arxiv.org/abs/2111.07991

```
@article{zhai2022lit,
  title={LiT: Zero-Shot Transfer with Locked-image Text Tuning},
  author={Zhai, Xiaohua and Wang, Xiao and Mustafa, Basil and Steiner, Andreas and Keysers, Daniel and Kolesnikov, Alexander and Beyer, Lucas},
  journal={CVPR},
  year={2022}
}
```

Model card:
https://github.com/google-research/vision_transformer/blob/main/model_cards/lit.md

Colabs:

- https://colab.research.google.com/github/google-research/vision_transformer/blob/main/lit.ipynb
- https://colab.research.google.com/github/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/lit.ipynb

### Results

| Model | Download link | ImageNet 0-shot | MS-COCO I→T | MS-COCO T→I | Config `arg` |
| :---  | :---:         | :---:           | :---:       | :---:       | :---         |
| mixed_L16L | [link](https://storage.googleapis.com/vit_models/lit/LiT-L16L.npz) | 75.7 | 48.5 | 31.2 | `txt=bert_large,img=L/16` |
| mixed_B16B | [link](https://storage.googleapis.com/vit_models/lit/LiT-B16B.npz) | 72.1 | 49.4 | 31.1 | `txt=bert_base,img=B/16,img_head` |
| mixed_B16B_2 | [link](https://storage.googleapis.com/vit_models/lit/LiT-B16B.npz) | 73.9 | 51.5 | 31.8 | `txt=bert_base,img=B/16` |
| coco_B16B | [link](https://storage.googleapis.com/vit_models/lit/big_vision/coco_B16B/checkpoint.npz) | 20.7 | 47.2 | 32.1 | `txt=bert_base,img=B/16` |

The first three rows are the best available models trained on open source data,
originally published in the [`google-research/vision_transformer`] repository.
These models were re-evaluated with this codebase using the following commands:

```bash
big_vision.tools.eval_only --config big_vision/configs/proj/image_text/lit_coco.py:txt=bert_base,img=B/16,img_head,init=gs://vit_models/lit/LiT-B16B.npz

big_vision.tools.eval_only --config big_vision/configs/proj/image_text/lit_coco.py:txt=bert_base,img=B/16_2,init=gs://vit_models/lit/LiT-B16B_2.npz

big_vision.tools.eval_only --config big_vision/configs/proj/image_text/lit_coco.py:txt=bert_large,img=L/16,init=gs://vit_models/lit/LiT-L16L.npz
```

Unfortunately, the public multi-modal datasets [`CC12M`] and [`YFCC100M`] are
not yet available in [`tfds`], so these models cannot be reproduced with the
codebase. For this reason we provide the much weaker model `coco_B16B` in the
third row, which was trained on the small `tfds` dataset [`coco_captions`], and
can be used to verify correctness of the codebase
([workdir](https://console.cloud.google.com/storage/browser/vit_models/lit/big_vision/coco_B16B/)).

[`google-research/vision_transformer`]: https://github.com/google-research/vision_transformer
[`CC12M`]: https://arxiv.org/abs/2102.08981
[`YFCC100M`]: https://arxiv.org/abs/1503.01817
[`tfds`]: https://www.tensorflow.org/datasets/api_docs/python/tfds
[`coco_captions`]: https://www.tensorflow.org/datasets/catalog/coco_captions


### Changelog

- 2022-08-18: Added LiT-B16B_2 model that was trained for 60k steps
  (LiT_B16B: 30k) without linear head on the image side (LiT_B16B: 768) and has
  better performance.
