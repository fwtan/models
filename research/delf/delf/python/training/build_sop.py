from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags

import numpy as np
import pandas as pd
import tensorflow as tf
import os.path as osp

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/tmp/', 'data directory.')
flags.DEFINE_string('out_dir', '/tmp/', 'Output data directory.')
flags.DEFINE_integer('num_shards', 128, 'Number of shards in output data.')



def _get_image_files_and_labels(data_dir, split):
    image_sourcepath  = osp.join(data_dir, 'images')
    files = pd.read_table(osp.join(data_dir, 'Info_Files/Ebay_%s.txt'%split), header=0, delimiter=' ')
    image_paths, file_ids, labels = [], [], []
    for key, img_id, img_path in zip(files['class_id'], files['image_id'], files['path']):
        image_paths.append(osp.join(image_sourcepath, img_path))
        file_ids.append(img_id-1)
        labels.append(key-1)
    return image_paths, file_ids, labels


def _process_image(filename):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.jpg'.

  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  Raises:
    ValueError: if parsed image has wrong number of dimensions or channels.
  """
  # Read the image file.
  with tf.io.gfile.GFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = tf.io.decode_jpeg(image_data, channels=3)
#   print(image.shape)

  # Check that image converted to RGB
  if len(image.shape) != 3:
    raise ValueError('The parsed image number of dimensions is not 3 but %d' %
                     (image.shape))
  height = image.shape[0]
  width = image.shape[1]
  if image.shape[2] != 3:
    raise ValueError('The parsed image channels is not 3 but %d' %
                     (image.shape[2]))

  return image_data, height, width


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(file_id, image_buffer, height, width, label=None):
  """Build an Example proto for the given inputs.

  Args:
    file_id: string, unique id of an image file, e.g., '97c0a12e07ae8dd5'.
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
    label: integer, the landmark id and prediction label.

  Returns:
    Example proto.
  """
  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'
  features = {
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace.encode('utf-8')),
      'image/channels': _int64_feature(channels),
      'image/format': _bytes_feature(image_format.encode('utf-8')),
      'image/id': _bytes_feature(str(file_id).zfill(16).encode('utf-8')),
      'image/encoded': _bytes_feature(image_buffer)
  }
  if label is not None:
    features['image/class/label'] = _int64_feature(label)
  example = tf.train.Example(features=tf.train.Features(feature=features))

  return example


def _write_tfrecord(output_prefix, image_paths, file_ids, labels):
  """Read image files and write image and label data into TFRecord files.

  Args:
    output_prefix: string, the prefix of output files, e.g. 'train'.
    image_paths: list of strings, the paths to images to be converted.
    file_ids: list of strings, the image unique ids.
    labels: list of integers, the landmark ids of images. It is an empty list
      when output_prefix='test'.

  Raises:
    ValueError: if the length of input images, ids and labels don't match
  """
#   if output_prefix == 'test':
#     labels = [None] * len(image_paths)
#   if not len(image_paths) == len(file_ids) == len(labels):
#     raise ValueError('length of image_paths, file_ids, labels shoud be the' +
#                      ' same. But they are %d, %d, %d, respectively' %
#                      (len(image_paths), len(file_ids), len(labels)))

  spacing = np.linspace(0, len(image_paths), FLAGS.num_shards + 1, dtype=np.int)

  for shard in range(FLAGS.num_shards):
    output_file = os.path.join(
        FLAGS.out_dir,
        '%s-%.5d-of-%.5d' % (output_prefix, shard, FLAGS.num_shards))
    writer = tf.io.TFRecordWriter(output_file)
    print('Processing shard ', shard, ' and writing file ', output_file)
    for i in range(spacing[shard], spacing[shard + 1]):
      image_buffer, height, width = _process_image(image_paths[i])
      example = _convert_to_example(file_ids[i], image_buffer, height, width,
                                    labels[i])
      writer.write(example.SerializeToString())
    writer.close()


def _build_tfrecord_dataset(split, data_dir):
  """Build a TFRecord dataset.

  Args:
    split: 'train' or 'test' to indicate which set of data to be processed.
    csv_path: path to the Google-landmark Dataset csv Data Sources files.
    image_dir: directory that stores downloaded images.

  Returns:
    Nothing. After the function call, sharded TFRecord files are materialized.
  """

  image_paths, file_ids, labels = _get_image_files_and_labels(data_dir, split)
  _write_tfrecord(split, image_paths, file_ids, labels)


def main(unused_argv):
  _build_tfrecord_dataset('train', FLAGS.data_dir)
  _build_tfrecord_dataset('test', FLAGS.data_dir)


if __name__ == '__main__':
  app.run(main)
