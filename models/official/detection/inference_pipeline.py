from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging
sys.path.insert(1, "/home/mihirgoyal/ds-video-commerce-poc/fashionpedia_pretrained_model/tpu/models/official/efficientnet")
sys.path.insert(2, "/home/mihirgoyal/ds-video-commerce-poc/fashionpedia_pretrained_model/tpu/models")
logger=logging.getLogger('logger')  
logger.setLevel(logging.DEBUG)

import cv2
import csv
import json
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_api
import tensorflow.compat.v1 as tf

from dataloader import mode_keys
from projects.fashionpedia.configs import factory as config_factory
from projects.fashionpedia.modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict
from time import perf_counter
from google.cloud import bigquery, storage
import argparse

def read_image(filename, image_s):
    img = cv2.imread(filename)
    img = cv2.resize(img, (image_s, image_s))
    cv2.imwrite(filename, img)
    with tf.gfile.GFile(filename, 'rb') as f:
        image_bytes = f.read()
    width, height = (image_s, image_s)
    return image_bytes, width, height

def get_predictions(predictions, image_bytes, width, height):
    predictions_np = sess.run(predictions, feed_dict={image_input: image_bytes})
    num_detections = int(predictions_np['num_detections'][0])
    np_boxes = predictions_np['detection_boxes'][0, :num_detections]
    np_scores = predictions_np['detection_scores'][0, :num_detections]
    np_classes = predictions_np['detection_classes'][0, :num_detections]
    np_classes = np_classes.astype(np.int32)
    np_attributes = predictions_np['detection_attributes'][0, :num_detections, :]
    np_masks = None
    if 'detection_masks' in predictions_np:
        instance_masks = predictions_np['detection_masks'][0, :num_detections]
        np_masks = mask_utils.paste_instance_masks(instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width)
        encoded_masks = [ mask_api.encode(np.asfortranarray(np_mask)) for np_mask in list(np_masks) ]
    return np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FKInference')
    parser.add_argument('--model_name', type=str, default='attribute_mask_rcnn')
    parser.add_argument('--path_to_checkpoint', type=str, default='projects/fashionpedia/checkpoints/fashionpedia-spinenet-143/model.ckpt')
    parser.add_argument('--path_to_config', type=str, default='projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml')
    parser.add_argument('--path_to_label_map', type=str, default='projects/fashionpedia/dataset/fashionpedia_label_map.csv')
    parser.add_argument('--path_to_attribute_map', type=str, default='projects/fashionpedia/dataset/attributes_label_map.json')
    parser.add_argument('--min_score_threshold', type=int, default=0.5)
    parser.add_argument('--path_to_catalog', type=str, default='FK_Inference/FK_catalog_04_01.csv')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--params_override', type=str, default='')
    args = parser.parse_args()
        
    logger.info('LOADING THE LABEL MAP')
    label_map_dict = {}
    with tf.gfile.Open(args.path_to_label_map, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=':')
        for row in reader:
            label_map_dict[int(row[0])] = { 'id': int(row[0]), 'name': row[1] }
    logger.info('DONE LOADING')

    logger.info('LOADING THE ATTRIBUTE MAP')
    with open(args.path_to_attribute_map, 'r') as attribute_file:
        data = json.load(attribute_file)
        attribute_index = data['attributes']
    logger.info('DONE LOADING')
    
    params = config_factory.config_generator(args.model_name)
    params = params_dict.override_params_dict(params, args.path_to_config, is_strict=True)
    params = params_dict.override_params_dict(params, args.params_override, is_strict=True)
    params.override({ 'architecture': { 'use_bfloat16': False } }, is_strict=True)
    params.validate()
    params.lock()

    model = model_factory.model_generator(params)
    tf.compat.v1.disable_eager_execution()
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [args.image_size, args.image_size]
    image, image_info = input_utils.resize_and_crop_image(image, image_size, image_size, aug_scale_min=1.0, aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    outputs = model.build_outputs(images, {'image_info': images_info}, mode=mode_keys.PREDICT)
    outputs['detection_boxes'] = ( outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]) )
    predictions = outputs
    image_with_detections_list = []

    saver = tf.train.Saver()
    sess = tf.Session()

    logger.info('LOADING THE CHECKPOINT')
    saver.restore(sess, args.path_to_checkpoint)
    logger.info('DONE LOADING')
    
    image_bytes, width, height = read_image('test.jpg', args.image_size)
    np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks = get_predictions(predictions, image_bytes, width, height)
    
    valid_ids = list(range(1,13)) + list(range(21,23))
    ids = [i for i in range(len(np_scores)) if np_scores[i] > args.min_score_threshold and np_classes[i] in valid_ids]
    classes = [label_map_dict[np_classes[i]]['name'] for i in ids]
    class_name = label_map_dict[np_classes[ids[0]]]['name']
    attributes = np_attributes[ids[0]]
    filtered_attributes = [attribute_index[j]['name'] for j in range(len(attributes)) if attributes[j] > args.min_score_threshold]
    print(class_name, filtered_attributes)