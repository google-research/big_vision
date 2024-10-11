import os
import keras
import keras_nlp
import numpy as np
import PIL
import requests
import io
import matplotlib
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

os.environ["KERAS_BACKEND"] = "jax"
os.environ["KAGGLE_USERNAME"] = "milliewu1"
os.environ["KAGGLE_KEY"] = "9005588500915e31a0bc757e9c53a3ed"

keras.config.set_floatx("bfloat16")

paligemma = keras_nlp.models.PaliGemmaCausalLM.from_preset("pali_gemma_3b_mix_224")
paligemma.summary()
# Function that reads an image from a given URL

def crop_and_resize(image, target_size):
    width, height = image.size
    source_size = min(image.size)
    left = width // 2 - source_size // 2
    top = height // 2 - source_size // 2
    right, bottom = left + source_size, top + source_size
    return image.resize(target_size, box=(left, top, right, bottom))

def read_image(url, target_size):
    contents = io.BytesIO(requests.get(url).content)
    image = PIL.Image.open(contents)
    image = crop_and_resize(image, target_size)
    image = np.array(image)
    # Remove alpha channel if neccessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def parse_bbox_and_labels(detokenized_output: str):
  matches = re.finditer(
      '<loc(?P<y0>\d\d\d\d)><loc(?P<x0>\d\d\d\d)><loc(?P<y1>\d\d\d\d)><loc(?P<x1>\d\d\d\d)>'
      ' (?P<label>.+?)( ;|$)',
      detokenized_output,
  )
  labels, boxes = [], []
  fmt = lambda x: float(x) / 1024.0
  for m in matches:
    d = m.groupdict()
    boxes.append([fmt(d['y0']), fmt(d['x0']), fmt(d['y1']), fmt(d['x1'])])
    labels.append(d['label'])
  return np.array(boxes), np.array(labels)

def display_boxes(image, boxes, labels, target_image_size):
  h, l = target_image_size
  fig, ax = plt.subplots()
  ax.imshow(image)
  for i in range(boxes.shape[0]):
      y, x, y2, x2 = (boxes[i]*h)
      width = x2 - x
      height = y2 - y
      # Create a Rectangle patch
      rect = patches.Rectangle((x, y),
                               width,
                               height,
                               linewidth=1,
                               edgecolor='r',
                               facecolor='none')
      # Add label
      plt.text(x, y, labels[i], color='red', fontsize=12)
      # Add the patch to the Axes
      ax.add_patch(rect)

  plt.show()

def display_segment_output(image, segment_mask, target_image_size):
  # Calculate scaling factors
  h, w = target_image_size
  x_scale = w / 64
  y_scale = h / 64

  # Create coordinate grids for the new image
  x_coords = np.arange(w)
  y_coords = np.arange(h)
  x_coords = (x_coords / x_scale).astype(int)
  y_coords = (y_coords / y_scale).astype(int)
  resized_array = segment_mask[y_coords[:, np.newaxis], x_coords]
  # Create a figure and axis
  fig, ax = plt.subplots()

  # Display the image
  ax.imshow(image)

  # Overlay the mask with transparency
  ax.imshow(resized_array, cmap='jet', alpha=0.5)


target_size = (224, 224)
# image_url = 'https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png'
image_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
image = read_image(image_url, target_size)
# matplotlib.pyplot.imsave(image, '/home/millie/ReNaAnalysis/big_vision/imgs/cat_cropped.jpg')

prompt = 'What is this species?\n'
# prompt = 'svar no hvor står kuen?'
# prompt = 'answer fr quelle couleur est le ciel?'
# prompt = 'responda pt qual a cor do animal?'
output = paligemma.generate(
    inputs={
        "images": image,
        "prompts": prompt,
    }
)
print(output)