# This is an example of using
# https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_pascal_tf_record.py
# The structure should be like PASCAL VOC format dataset
# +Dataset
#   +Annotations
#   +JPEGImages
# python create_tfrecords_from_xml.py --image_dir=dataset/JPEGImages
#                                      --annotations_dir=dataset/Annotations
#                                      --label_map_path=object-detection.pbtxt
#                                      --output_path=data.record

import hashlib
import io
import logging
import os
import json
from pprint import pprint

from lxml import etree
import PIL.Image
import tensorflow as tf
import numpy as np
import cv2

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('image_dir', None, 'Path to image directory.')
flags.DEFINE_string('annotations_dir', None, 'Path to annotations directory.')
flags.DEFINE_string('output_path', None, 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'label_map.pbtxt', 'Path to label map proto')
flags.DEFINE_string('class_implications', None, 'JSON object literal mapping target classes to replacement classes')
flags.DEFINE_string('class_priorities', None, 'JSON array literal that defines class priorities from the least specific (first) to the most specific (last)')
flags.DEFINE_string('base_class', 'person', 'Use a two-class model, converting additional classes by using built-in implication rules (only applies if the class_implications paramted is not provided)')
flags.DEFINE_string('target_class', None, 'Use a two-class model, converting additional classes by using built-in implication rules (only applies if the class_implications paramted is not provided)')
flags.DEFINE_boolean('grayscale', False, 'Convert images to grayscale before packing into the TFRecord file')
flags.DEFINE_boolean('clahe', False, 'Use the CLAHE algorithm to improve contrast (only applies if grayscale is enabled)')
FLAGS = flags.FLAGS

DEFAULT_CLASS_PRIORITIES = ['person']

stats = {'impl_classes_replaced': 0, 'impl_images_replaced': 0}
clahe = None


def convertToJpeg(im):
    with io.BytesIO() as f:
        im.save(f, format='JPEG')
        return f.getvalue()


def load_image__PIL(full_path):
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    image = PIL.Image.open(io.BytesIO(encoded_jpg))

    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    if FLAGS.grayscale:
        image = image.convert('L')
        encoded_jpg = convertToJpeg(image)

    return image, encoded_jpg


def load_image__cv2(full_path):
    global clahe

    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()

    np_buffer = np.fromstring(encoded_jpg, np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_GRAYSCALE if FLAGS.grayscale else cv2.IMREAD_COLOR)

    if FLAGS.grayscale:
        if FLAGS.clahe:
            clahe = clahe or cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            image = clahe.apply(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        success, encoded_jpg = cv2.imencode('.jpg', image)
        if not success:
            raise RuntimeError('Failed to encode "{}"'.format(full_path))
        encoded_jpg = encoded_jpg.tobytes()

    return image, encoded_jpg


def dict_to_tf_example(data, image_dir, label_map_dict, class_impl=None):
    """Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: dict holding XML fields for a single image (obtained by
          running dataset_util.recursive_parse_xml_to_dict)
        image_dir: Path to image directory.
        label_map_dict: A map from string label names to integers ids.

    Returns:
        example: The converted tf.Example.
    """
    full_path = os.path.join(image_dir, data['filename'])
    # image, encoded_jpg = load_image__PIL(full_path)
    image, encoded_jpg = load_image__cv2(full_path)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    try:
        for obj in data['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
    except KeyError as ex:
        print(data['filename'] + ' without objects!')

    if class_impl is not None:
        assert len(classes) == len(classes_text)
        num_replaced = 0

        for i in range(len(classes)):
            implied_class = class_impl.get(classes_text[i].decode('utf-8'))
            if implied_class is None: continue
            classes[i] = label_map_dict[implied_class]
            classes_text[i] = implied_class.encode('utf-8')
            num_replaced += 1

        if num_replaced > 0:
            stats['impl_classes_replaced'] += num_replaced
            stats['impl_images_replaced'] += 1

    difficult_obj = [0]*len(classes)
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj)
    }))
    return example


def derive_implications(classes, class_1, class_2):
    prio = { label: index for index, label in enumerate(classes)}

    # Swap classes if their priorities are in the wrong order
    if prio[class_2] < prio[class_1]:
        class_1, class_2 = class_2, class_1

    class_1_prio = prio[class_1]
    class_2_prio = prio[class_2]

    # Create implications
    impl = {}
    for label in classes:
        curr_prio = prio[label]
        if curr_prio > class_2_prio:
            impl[label] = class_2
        elif curr_prio < class_2_prio and curr_prio != class_1_prio:
            impl[label] = class_1
    return impl


def main(_):
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)
    class_impl = json.loads(FLAGS.class_implications.replace("'", '"')) if FLAGS.class_implications else None
    class_prio = json.loads(FLAGS.class_priorities.replace("'", '"')) if FLAGS.class_priorities else DEFAULT_CLASS_PRIORITIES
    assert isinstance(class_prio, list)

    def has_class_arg(target_class):
        return target_class is not None and target_class in class_prio

    if class_impl is None and has_class_arg(FLAGS.base_class) and has_class_arg(FLAGS.target_class):
        class_impl = derive_implications(class_prio, FLAGS.base_class, FLAGS.target_class)
        pprint(class_impl)

    image_dir = FLAGS.image_dir
    assert image_dir is not None and os.path.isdir(image_dir)
    annotations_dir = FLAGS.annotations_dir or os.path.join(FLAGS.image_dir, "annotations")
    assert annotations_dir is not None and os.path.isdir(annotations_dir)
    assert FLAGS.output_path is not None and (not os.path.exists(FLAGS.output_path) or os.path.isfile(FLAGS.output_path))
    logging.info('Reading from dataset: ' + annotations_dir)
    examples_list = os.listdir(annotations_dir)

    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    stats['impl_classes_replaced'] = 0
    stats['impl_images_replaced'] = 0

    for idx, example in enumerate(examples_list):
        if example.endswith('.xml'):
            if idx % 50 == 0:
                print('On image %d of %d' % (idx, len(examples_list)))

            path = os.path.join(annotations_dir, example)
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, image_dir, label_map_dict, class_impl)
            writer.write(tf_example.SerializeToString())

    writer.close()
    if class_impl is not None:
        print("Replaced {} classes in {} images with implied classes".format(stats['impl_classes_replaced'],
                                                                             stats['impl_images_replaced']))


if __name__ == '__main__':
    tf.app.run()


# Import needed variables from tensorflow
# From tensorflow/models/research/
# protoc object_detection/protos/*.proto --python_out=.
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# python object_detection/builders/model_builder_test.py
