from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import numpy as np
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.platform import app
from delf import delf_config_pb2
from delf import feature_io
from delf import utils
from delf.python.detect_to_retrieve import dataset
from delf import extractor

cmd_args = None


def read_file(filename):
    with open(filename) as f:
        lines = f.read().splitlines()
    return lines


# Extensions.
_DELF_EXTENSION = '.delf'
# _IMAGE_EXTENSION = '.jpg'


# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def main(argv):
  if len(argv) > 1:
    raise RuntimeError('Too many command-line arguments.')
  print('Reading list of images from dataset file...')
  image_list = read_file(cmd_args.dataset_file_path)
  num_images = len(image_list)
  print(f'done! Found {num_images} images')

  # Parse DelfConfig proto.
  config = delf_config_pb2.DelfConfig()
  with tf.io.gfile.GFile(cmd_args.delf_config_path, 'r') as f:
    text_format.Merge(f.read(), config)

  # Create output directory if necessary.
  if not tf.io.gfile.exists(cmd_args.output_features_dir):
    tf.io.gfile.makedirs(cmd_args.output_features_dir)

  extractor_fn = extractor.MakeExtractor(config)

  start = time.time()
  for i in range(num_images):
    if i == 0:
      print('Starting to extract features...')
    elif i % _STATUS_CHECK_ITERATIONS == 0:
      elapsed = (time.time() - start)
      print('Processing image %d out of %d, last %d '
            'images took %f seconds' %
            (i, num_images, _STATUS_CHECK_ITERATIONS, elapsed))
      start = time.time()
    line = image_list[i]
    image_path, image_label, image_width, image_height = line.split(',')
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    input_image_filename = os.path.join(cmd_args.images_dir, image_path)
    output_feature_filename = os.path.join(cmd_args.output_features_dir, image_name + _DELF_EXTENSION)
    if tf.io.gfile.exists(output_feature_filename):
      print(f'Skipping {image_name}')
      continue
    
    im = np.array(utils.RgbLoader(input_image_filename))

    # Extract and save features.
    extracted_features = extractor_fn(im)
    locations_out = extracted_features['local_features']['locations']
    descriptors_out = extracted_features['local_features']['descriptors']
    feature_scales_out = extracted_features['local_features']['scales']
    attention_out = extracted_features['local_features']['attention']

    feature_io.WriteToFile(output_feature_filename, locations_out,
                           feature_scales_out, descriptors_out,
                           attention_out)
    

#   elapsed = (time.time() - start)
#   print('Processed %d query images in %f seconds' % (num_images, elapsed))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--delf_config_path',
      type=str,
      default='/tmp/delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--dataset_file_path',
      type=str,
      default='/tmp/gnd_roxford5k.mat',
      help="""
      Dataset file for Revisited Oxford or Paris dataset, in .mat format.
      """)
  parser.add_argument(
      '--images_dir',
      type=str,
      default='/tmp/images',
      help="""
      Directory where dataset images are located, all in .jpg format.
      """)
  parser.add_argument(
      '--output_features_dir',
      type=str,
      default='/tmp/features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
