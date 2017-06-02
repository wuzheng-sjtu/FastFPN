from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import zipfile
import re
import time
import numpy as np
import tensorflow as tf
from six.moves import urllib
from PIL import Image
import skimage.io as io
from matplotlib import pyplot as plt

#from libs.datasets.pycocotools.coco import COCO
#from libs.datasets.citytools import City
from tensorflow.python.lib.io.tf_record import TFRecordCompressionType
from libs.logs.log import LOG


FLAGS = tf.app.flags.FLAGS

#def _real_id_to_cat_id(catId):
#  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
#  real_id_to_cat_id = \
#    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16, 16: 17,
#     17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32, 29: 33, 30: 34,
#     31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46, 42: 47, 43: 48, 44: 49,
#     59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76, 68: 77, 69: 78, 70: 79, 71: 80, 72: 81,
#     73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}
#  return real_id_to_cat_id[catId]

#def _cat_id_to_real_id(readId):
#  """Note coco has 80 classes, but the catId ranges from 1 to 90!"""
#  cat_id_to_real_id = \
#    {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
#     18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30,
#     35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44,
#     50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58,
#     64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
#     82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
#  return cat_id_to_real_id[readId]
  
def cls2id(cls):
  clsdict = {'person':1, 'rider':2, 'car':3,'truck':4,'bus':5,'caravan':6, \
      'trailer':7,'train':8,'motorcycle':9,'bicycle':10}
  return clsdict[cls]

class ImageReader(object):
  def __init__(self):
    self._decode_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_data, channels=3)
    self._decode_png = tf.image.decode_png(self._decode_data)

  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_data: image_data})
    #assert len(image.shape) == 3
    #assert image.shape[2] == 1
    return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, num_shards):
  output_filename = 'city_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, num_shards)
  return os.path.join(dataset_dir, output_filename)


def _get_image_filenames(image_dir):
  return sorted(os.listdir(image_dir))


def _int64_feature(values):
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def _to_tfexample(image_data, image_format, label_data, label_format, height, width):
  """Encode only masks """
  return tf.train.Example(features=tf.train.Features(feature={
      #have doubts on 'encoded' here 
      #'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'label/encoded': _bytes_feature(label_data),
      'label/format': _bytes_feature(label_format),
      'label/height': _int64_feature(height),
      'label/width': _int64_feature(width),
  }))

def _to_tfexample_city(image_data, image_format, label_format,
                       height, width,
                       num_instances, gt_boxes):
  
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': _bytes_feature(image_data),
      'image/format': _bytes_feature(image_format),
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
  
      'label/num_instances': _int64_feature(num_instances), # N
      'label/gt_boxes': _bytes_feature(gt_boxes), # of shape (N, 5), (x1, y1, x2, y2, classid)
      #'label/gt_masks': _bytes_feature(masks),       # of shape (N, height, width)
    
      #'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
      'label/format': _bytes_feature(label_format),
  }))


def _to_tfexample_city_raw(image_id, image_data, 
                           height, width,
                           num_instances, gt_boxes):
  """ just write a raw input"""
  return tf.train.Example(features=tf.train.Features(feature={
    'image/img_id': _int64_feature(image_id),
    'image/encoded': _bytes_feature(image_data),
    'image/height': _int64_feature(height),
    'image/width': _int64_feature(width),
    'label/num_instances': _int64_feature(num_instances),  # N
    'label/gt_boxes': _bytes_feature(gt_boxes),  # of shape (N, 5), (x1, y1, x2, y2, classid)
    #'label/gt_masks': _bytes_feature(masks),  # of shape (N, height, width)
    #'label/encoded': _bytes_feature(label_data),  # deprecated, this is used for pixel-level segmentation
  }))


def _get_city_boxes(img_dir, img_name):
  """ get the masks for all the instances
  Note: some images are not annotated
  Return:
    classes, mx1
    bboxes, mx4
  """
  classes = []
  bboxes = []
  img_name = img_name.split('/')[3]
  filename = os.path.splitext(img_name)[0][:-12]+'.txt'
  anno_name = os.path.join(img_dir, 'annotations', filename)
  with open(anno_name, 'r') as f:
    data = f.read()
  objs = re.findall(r'\'label\'\:\'(.*?)\'\s\'bounding\sbox\'\:\((.*?),(.*?),(.*?),(.*?)\)', data)
  for ix, obj in enumerate(objs):
            # Make pixel indexes 0-based
    coor = obj[1:]
         #coor = re.findall('\d+', obj)
    x1 = float(coor[0])
    y1 = float(coor[1])
    x2 = float(coor[2])
    y2 = float(coor[3])
    clsid = cls2id(obj[0])
    box = [x1, y1, x2, y2]
    classes.append(clsid)
    bboxes.append(box)

  classes = np.asarray(classes)
  bboxes = np.asarray(bboxes)
  # to x1, y1, x2, y2
  if bboxes.shape[0] <= 0:
    bboxes = np.zeros([0, 4], dtype=np.float32)
    classes = np.zeros([0], dtype=np.float32)
    print ('None Annotations %s' % img_name)
#    LOG('None Annotations %s' % img_name)

  gt_boxes = np.hstack((bboxes, classes[:, np.newaxis]))
  gt_boxes = gt_boxes.astype(np.float32)
  
  return gt_boxes
  
global imgs
imgs = []
for datasplit in ['train', 'val']:
    split_dir = os.path.join(FLAGS.dataset_dir, datasplit)
    imgs.extend(sorted(os.listdir(split_dir)))


def _add_to_tfrecord(record_dir, image_dir, annotation_dir, split_name):
  """Loads image files and writes files to a TFRecord.
  Note: masks and bboxes will lose shape info after converting to string.
  """

  #assert split_name in ['train2014', 'val2014', 'valminusminival2014', 'minival2014']
  #annFile = os.path.join(annotation_dir, 'instances_%s.json' % (split_name))
  
  #city = City()

  #cats = coco.loadCats(coco.getCatIds())
  #print ('%s has %d images' %(split_name, len(coco.imgs)))
  #imgs = [(img_id, coco.imgs[img_id]) for img_id in coco.imgs]
  #idx2img = { for i in os.listdir()}
  # construct the index to images
  #we use it to get the index of each image name

  num_imgs = len(os.listdir(os.path.join(image_dir, split_name)))
  
  num_shards = int(num_imgs / 2500)
  num_per_shard = int(math.ceil(num_imgs / float(num_shards)))
  
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    image_reader = ImageReader()
    
    # encode mask to png_string
    #mask_placeholder = tf.placeholder(dtype=tf.uint8)
    #encoded_image = tf.image.encode_png(mask_placeholder)
    
    with tf.Session('') as sess:
      for shard_id in range(num_shards):
        record_filename = _get_dataset_filename(record_dir, split_name, shard_id, num_shards)
        options = tf.python_io.TFRecordOptions(TFRecordCompressionType.ZLIB)
        with tf.python_io.TFRecordWriter(record_filename, options=options) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, num_imgs)
          for i in range(start_ndx, end_ndx):
            if i % 50 == 0:
                sys.stdout.write('\r>> Converting image %d/%d shard %d\n' % (
                  i + 1, num_imgs, shard_id))
                sys.stdout.flush()
            
            # image id and path
            img_name = sorted(os.listdir(os.path.join(image_dir, split_name)))[i]
	    img_id = imgs.index(img_name)
            #split = img_name.split('_')[1]
            img_name = os.path.join(image_dir, split_name, img_name)
            
            if FLAGS.vis:
              im = Image.open(img_name)
              im.save('img.png')
              plt.figure(0)
              plt.axis('off')
              plt.imshow(im)
              # plt.show()
              # plt.close()
            
            # jump over the damaged images
            #if str(img_id) == '320612':
            #  continue
            
            # process anns
            height, width = 1024, 2048 #size of cityscapes
            gt_boxes = _get_city_boxes(image_dir, img_name)

            # read image as RGB numpy
            img = np.array(Image.open(img_name))
            if img.size == height * width:
                print ('Gray Image %s' % str(img_id))
                im = np.empty((height, width, 3), dtype=np.uint8)
                im[:, :, :] = img[:, :, np.newaxis]
                img = im

            img = img.astype(np.uint8)
            assert img.size == width * height * 3, '%s' % str(img_id)

            img_raw = img.tostring()
            #mask_raw = mask.tostring()
            
            example = _to_tfexample_city_raw(
              img_id,
              img_raw,
              height, width, gt_boxes.shape[0],
              gt_boxes.tostring())
            
            tfrecord_writer.write(example.SerializeToString())
  sys.stdout.write('\n')
  sys.stdout.flush()



def run(dataset_dir, dataset_split_name='train'):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  # for url in _DATA_URLS:
  #   download_and_uncompress_zip(url, dataset_dir)

  record_dir     = os.path.join(dataset_dir, 'records')
  annotation_dir = os.path.join(dataset_dir, 'annotations')

  if not tf.gfile.Exists(record_dir):
    tf.gfile.MakeDirs(record_dir)

  # process the training, validation data:
  if dataset_split_name in ['train', 'val']:
      _add_to_tfrecord(record_dir,
                       dataset_dir,
                       annotation_dir,
                       dataset_split_name)

  
  print('\nFinished converting the city dataset!')
