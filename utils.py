import os
import urllib
import tarfile
from io import BytesIO
from base64 import b64encode

import functools
from matplotlib import gridspec
import matplotlib.pylab as plt
from PIL import Image
import numpy as np
import tensorflow as tf


def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(img_file_buffer, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  image = Image.open(img_file_buffer)
  img_array = np.array(image)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = img_array.astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w  * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  return plt

def get_image_download_link(image_tensor):
  """
    Generates a link allowing the PIL image to be downloaded
      input:  image tensor
      output: href string
  """
  image_array = image_tensor.numpy()[0] * 255
  img = Image.fromarray(image_array.astype(np.uint8))
  buffered = BytesIO()
  img.save(buffered, format="JPEG")
  img_str = b64encode(buffered.getvalue()).decode()
  href = '<a href="data:file/jpg;base64,{}" download="stylized.jpeg">Download stylized image</a>'.format(img_str)
  return href
  

