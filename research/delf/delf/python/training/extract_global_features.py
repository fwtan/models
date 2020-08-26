import os, time
import os.path as osp
import numpy as np
from pathlib import Path
import tensorflow as tf
from PIL import Image
from scipy.spatial import distance
import argparse, sys
from delf import datum_io
from tensorflow.python.platform import app


REQUIRED_SIGNATURE = 'serving_default'
REQUIRED_OUTPUT = 'global_descriptor'
embedding_fn = None


def get_embedding(image_path: Path) -> np.ndarray:
    image_data = np.array(Image.open(str(image_path)).convert('RGB'))
    image_tensor = tf.convert_to_tensor(image_data)
    return embedding_fn(image_tensor)[REQUIRED_OUTPUT].numpy()


class Submission:
    def __init__(self, name, model):
        self.name = name
        self.model = model
    
    def get_id(self, image_path: Path):
        return osp.splitext(osp.basename(image_path))[0]

    def get_embeddings(self, image_paths):
        embeddings = [get_embedding(image_path) for i, image_path in enumerate(image_paths)]
        ids = [self.get_id(image_path) for image_path in image_paths]
        return ids, embeddings


def load_model(saved_model_path):
    model = tf.saved_model.load(str(saved_model_path))
    found_signatures = list(model.signatures.keys())
    if REQUIRED_SIGNATURE not in found_signatures:
        return None
    outputs = model.signatures[REQUIRED_SIGNATURE].structured_outputs
    if REQUIRED_OUTPUT not in outputs:
        return None
    global embedding_fn
    embedding_fn = model.signatures[REQUIRED_SIGNATURE]
    return Submission('delf', model)



cmd_args = None
# Extension of feature files.
_DELF_EXT = '.delf'
# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.io.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


def main(unused_argv):
  print('Reading list of images...')
  image_paths = _ReadImageList(cmd_args.list_images_path)
  num_images = len(image_paths)
  print(f'done! Found {num_images} images')

  print('Loading model...')
  extractor = load_model(cmd_args.model_path)
  ids, embeddings = extractor.get_embeddings(image_paths)
  for i in range(len(ids)):
      image_name = ids[i]
      global_descriptor = embeddings[i]
      output_global_feature_filename = osp.join(cmd_args.output_dir, image_name + _DELF_EXT)
      datum_io.WriteToFile(global_descriptor, output_global_feature_filename)
      if i % _STATUS_CHECK_ITERATIONS == 0:
          print(i, image_name, global_descriptor.shape)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument('--model_path', type=str, default=None, 
    help="""tensorflow SavedModel""")
  parser.add_argument('--list_images_path', type=str, default='list_images.txt', 
    help="""Path to list of images whose DELF features will be extracted.""")
  parser.add_argument('--output_dir', type=str, default='test_features',
    help="""Directory where DELF features will be written to. Each image's features will be written to a file with same name, and extension replaced by .delf.""")
  cmd_args, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)