import PIL
import keras
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding
import os

import keras_nlp
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import requests
import io
import kagglehub
import tensorflow as tf

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

backend = jax.lib.xla_bridge.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")



path = "/big_vision/big_vision/configs/proj/paligemma/ckpts/3b_mix_224.npz"

# Function that reads an image from a given URL

def read_image(url):
    contents = io.BytesIO(requests.get(url).content)
    image = Image.open(contents).convert("RGB")
    image = image.resize((224, 224))  # Resize to the model input size
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = image.astype(np.float32)  # Ensure the dtype is float32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load an image
image_url = 'https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png'
image = read_image(image_url)


pali_gemma_lm = keras_nlp.models.PaliGemmaCausalLM.from_preset(path)
# Set the backend and load the model
os.environ["KERAS_BACKEND"] = "jax"
keras.config.set_floatx("float32")

# Use a generate call with a single image and prompt
prompt = 'answer en where is the cow standing?\n'
output = pali_gemma_lm.generate(
    inputs={
        "images": image,
        "prompts": prompt,
    }
)
print(output)
