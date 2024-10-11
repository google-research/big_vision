```
export KAGGLE_USERNAME=milliewu1
export KAGGLE_KEY=9005588500915e31a0bc757e9c53a3ed

# See https://www.kaggle.com/models/google/paligemma for a full list of models.
export MODEL_NAME=paligemma-3b-mix-224
export CKPT_FILE=paligemma-3b-mix-224.npz

mkdir ckpts/
cd ckpts/

curl -L -u $KAGGLE_USERNAME:$KAGGLE_KEY\
  -o 3b_mix_224.npz \
  https://www.kaggle.com/api/v1/models/google/paligemma/jax/$MODEL_NAME/1/download/$CKPT_FILE
```