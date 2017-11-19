#!/usr/bin/env python

import glob
import random
import tensorflow as tf
import os
import cv2

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(example, MIN_WIDTH=2, MAX_WIDTH=200, MIN_HEIGHT=2, MAX_HEIGHT=400):
    ss = example.split(' ')

    if len(ss) < 2:
        raise ValueError('Invalid example:' + example)

    path = ss[0]
    img = cv2.imread(path)
    if img is None:
        raise ValueError('Missed image:' + path)

    height, width, depth = img.shape

    with tf.gfile.GFile(path, 'rb') as fid:
        encoded_image = fid.read()

    format = path.split('.')[-1]
    image_format = format.encode()
    enc_path = path.encode()

    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box
    # (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box
    # (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)

    num_boxes = int(ss[1])
    for i in range(0, num_boxes):
        n = i * 4 + 2
        x_min = int(ss[n])
        y_min = int(ss[n + 1])
        x_max = int(ss[n + 2])
        y_max = int(ss[n + 3])

        if x_min >= x_max:
            x_min, x_max = x_max, x_min
        if y_min >= y_max:
            y_min, y_max = y_max, y_min

        width = x_max - x_min
        height = y_max - y_min

        if x_max >= width:
            x_max = width - 1
        if y_max > height:
            y_max = height - 1

        if width < MIN_WIDTH:
            raise ValueError('Box width smaller than min (' + str(width) + 'x' + str(height) + ') at ' + example)
        if width > MAX_WIDTH:
            raise ValueError('Box width bigger than max (' + str(width) + 'x' + str(height) + ') at ' + example)
        if height < MIN_HEIGHT:
            raise ValueError('Box height smaller than min (' + str(width) + 'x' + str(height) + ') at ' + example)
        if height > MAX_HEIGHT:
            raise ValueError('Box height bigger than max (' + str(width) + 'x' + str(height) + ') at ' + example)

        xmins.append(float(x_min / width))
        xmaxs.append(float(x_max / width))
        ymins.append(float(y_min / height))
        ymaxs.append(float(y_max / height))
        classes_text.append('tl'.encode())
        classes.append(int(1))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(enc_path),
        'image/source_id': bytes_feature(enc_path),
        'image/encoded': bytes_feature(encoded_image),
        'image/format': bytes_feature(image_format),
        'image/object/bbox/xmin': float_list_feature(xmins),
        'image/object/bbox/xmax': float_list_feature(xmaxs),
        'image/object/bbox/ymin': float_list_feature(ymins),
        'image/object/bbox/ymax': float_list_feature(ymaxs),
        'image/object/class/text': bytes_list_feature(classes_text),
        'image/object/class/label': int64_list_feature(classes),
    }))

    return tf_example


def get_all_labels(annotation='*/train*.txt'):
    lines = []
    for file in glob.glob(annotation):
        with open(file) as f:
            lines.extend(f.read().splitlines())
    return lines

def create_tf_record(record_filename, examples):
    writer = tf.python_io.TFRecordWriter(record_filename)

    counter = 1
    len_examples = len(examples)
    step = len_examples / 100
    i = 1
    for example in examples:
        try:
            tf_example = create_tf_example(example)
            writer.write(tf_example.SerializeToString())
        except ValueError as err:
            print(err.args)

        if counter > step:
            print("Percent done", i)
            i += 1
            counter = 0
        else:
            counter += 1

    writer.close()

def main(_):
    dir = 'tf_records_'+FLAGS.set
    if not os.path.exists(dir):
        os.makedirs(dir)
    train_output_path = os.path.join(dir, 'train.record')
    val_output_path = os.path.join(dir, 'val.record')
    path = '/data/traffic_lights/{}/train*.txt'.format(FLAGS.set)
    print('using annotatilns from', path)	
    examples = get_all_labels(path)
    #examples = examples[:10]  # for testing
    len_examples = len(examples)
    print("Loaded ", len(examples), "examples")

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples)
    num_train = int(0.7 * len_examples)
    train_examples = examples[:num_train]
    val_examples = examples[num_train:]
    print('%d training and %d validation examples.', len(train_examples), len(val_examples))

    print('Creating training record...')
    create_tf_record(train_output_path, train_examples)
    print('Creating validation record...')
    create_tf_record(val_output_path, val_examples)

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('set','','')
#    flags.DEFINE_string('input_examples', '/data/traffic_lights/*/train*.txt', 'Path to examples')
#    flags.DEFINE_string('output_dir', './tf_records', 'Path to output TFRecord')
    FLAGS = flags.FLAGS

    with tf.device('/device:GPU:0'):
    	tf.app.run()
