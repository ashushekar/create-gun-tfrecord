from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random

from lxml import etree
import PIL.Image
import tensorflow as tf
import contextlib2

import label_map_util
from models.research.object_detection.utils import dataset_util
from models.research.object_detection.dataset_tools import tf_record_creation_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', 'YTO_multi', 'Root directory to raw gun dataset.')
flags.DEFINE_string('set', 'all', 'Convert training set, validation set or '
                    'merged set.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_dir', 'output', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/yto_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
flags.DEFINE_integer('num_shards', 10, 'Number of TFRecord shards')
FLAGS = flags.FLAGS

SETS = ['train', 'val', 'trainval', 'test', 'all']


def dict_to_tf_example(data,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(data['folder'], image_subdirectory, data['filename'])
  full_path = os.path.join(dataset_directory, img_path)
  with tf.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(label_map_dict[obj['name']])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses)
  }))
  print(data['filename'])
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     annotations_dir,
                     examples,
                     image_dir=FLAGS.data_dir,
                     faces_only=False):
  """Creates a TFRecord file from examples.
  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in enumerate(examples):
        if idx % 100 == 0:
            logging.info('On image %d of %d', idx, len(examples))
        path = os.path.join(annotations_dir, example + '.xml')
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        try:
            tf_example = dict_to_tf_example(data, FLAGS.data_dir, label_map_dict,
                                          FLAGS.ignore_difficult_instances)
            if tf_example:
                shard_idx = idx % num_shards
                output_tfrecords[shard_idx].write(tf_example.SerializeToString())
        except ValueError:
            logging.warning('Invalid example: %s, ignoring.', path)


def main(_):
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    data_dir = FLAGS.data_dir

    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    examples_path = os.path.join(data_dir, FLAGS.set + 'YTO.txt')

    annotations_dir = os.path.join(data_dir, FLAGS.annotations_dir)
    examples_list = dataset_util.read_examples_list(examples_path)

    random.seed(42)

    random.shuffle(examples_list)
    num_examples = len(examples_list)
    print("Total Records in Dataset: {}".format(num_examples))
    num_train = int(0.8 * num_examples)
    train_examples = examples_list[:num_train]
    test_examples = examples_list[num_train:]
    train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
    test_output_path = os.path.join(FLAGS.output_dir, 'test.record')
    print("Training Set conversion")
    create_tf_record(
        train_output_path,
        FLAGS.num_shards,
        label_map_dict,
        annotations_dir,
        train_examples,
        image_dir=FLAGS.data_dir)

    print("Test Set conversion")
    create_tf_record(
        test_output_path,
        FLAGS.num_shards,
        label_map_dict,
        annotations_dir,
        test_examples,
        image_dir=FLAGS.data_dir)


if __name__ == '__main__':
  tf.app.run()
