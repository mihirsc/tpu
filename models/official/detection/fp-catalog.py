from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import logging

sys.path.append('../efficientnet')
sys.path.append('../..')
sys.path.append(os.path.join(os.path.dirname(__file__), "query-dataset-gen-tool", "util"))
sys.path.append(os.path.join(os.path.dirname(__file__), "query-dataset-gen-tool", "service"))
sys.path.append(os.path.join(os.path.dirname(__file__), "query-dataset-gen-tool", "temp"))

logger=logging.getLogger('logger')  
logger.setLevel(logging.DEBUG)

from data_util import read_catalog
from tag_annotation_service import add_tags
from cos_sim import match_items
from parse_json import parse_json

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
from google.cloud import bigquery, storage

from search_utils.color_detector import get_dominant_color
from search_utils.attribute_cleanup import clean_attributes
from search_utils.map_fashionpedia_to_flipkart import map_fashionpedia2flipkart as process_tags

class Model:
    min_score_threshold = 0.8
    image_size = 640
    
    def __init__(self,
                 model_name='attribute_mask_rcnn',
                 path_to_checkpoint='projects/fashionpedia/checkpoints/fashionpedia-spinenet-143/model.ckpt',
                 path_to_config='projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml',
                 path_to_label_map='projects/fashionpedia/dataset/fashionpedia_label_map.csv',
                 path_to_attribute_map='projects/fashionpedia/dataset/attributes_label_map.json',
                 params_override=''):

        self.label_map_dict, self.attribute_index = self.getDatasetInfo(path_to_label_map, path_to_attribute_map)

        params = config_factory.config_generator(model_name)
        params = params_dict.override_params_dict(params, path_to_config, is_strict=True)
        params = params_dict.override_params_dict(params, params_override, is_strict=True)
        params.override({ 'architecture': { 'use_bfloat16': False } }, is_strict=True)
        params.validate()
        params.lock()

#         tf.compat.v1.disable_eager_execution() #For TF2
        model = model_factory.model_generator(params)
        self.image_input = tf.placeholder(shape=(), dtype=tf.string)
        image = tf.io.decode_image(self.image_input, channels=3)
        image.set_shape([None, None, 3])

        image = input_utils.normalize_image(image)
        image_s = [self.image_size, self.image_size]
        image, image_info = input_utils.resize_and_crop_image(image, image_s, image_s, aug_scale_min=1.0, aug_scale_max=1.0)
        image.set_shape([image_s[0], image_s[1], 3])
        images = tf.reshape(image, [1, image_s[0], image_s[1], 3])
        images_info = tf.expand_dims(image_info, axis=0)

        outputs = model.build_outputs(images, {'image_info': images_info}, mode=mode_keys.PREDICT)
        outputs['detection_boxes'] = ( outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]) )
        self.predictions = outputs

        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, path_to_checkpoint)
        
        self.catalog_data = read_catalog()
        add_tags(self.catalog_data)

    def getDatasetInfo(self, path_to_label_map, path_to_attribute_map):
        label_map_dict = {}
        with tf.gfile.Open(path_to_label_map, 'r') as csv_file:
            reader = csv.reader(csv_file, delimiter=':')
            for row in reader:
                label_map_dict[int(row[0])] = { 'id': int(row[0]), 'name': row[1] }
        
        attribute_index = None
        with open(path_to_attribute_map, 'r') as attribute_file:
            data = json.load(attribute_file)
            attribute_index = data['attributes']
        return label_map_dict, attribute_index

    def read_image(self, filename, image_s):
        img = cv2.imread(filename)
        img = cv2.resize(img, (image_s, image_s))
        cv2.imwrite(filename, img)
        with tf.gfile.GFile(filename, 'rb') as f:
            image_bytes = f.read()
        width, height = (image_s, image_s)
        return image_bytes, width, height

    def get_predictions(self, image_bytes, width, height):
        predictions_np = self.sess.run(self.predictions, feed_dict={self.image_input: image_bytes})
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
    
    def post_process_predictions(self, np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks):
        predictions = {}
        for i in range(len(np_classes)):
            class_id = np_classes[i]
            class_name = self.label_map_dict[class_id]['name']
            confidence = np_scores[i]
            attributes = [self.attribute_index[j] for j in range(len(np_attributes[i])) if np_attributes[i][j] > self.min_score_threshold]
            attribute_scores = [np_attributes[i][j] for j in range(len(np_attributes[i])) if np_attributes[i][j] > self.min_score_threshold]
            attributes_class = {}
            for j in range(len(attributes)):
                name = attributes[j]['name']
                score = attribute_scores[j]
                supername = attributes[j]['supercategory']
                if supername not in attributes_class:
                    attributes_class[supername] = {'confidence': 0, 'name': ''}
                if attributes_class[supername]['confidence'] <= score:
                    attributes_class[supername]['name'] = name
                    attributes_class[supername]['confidence'] = score
            
            if confidence < self.min_score_threshold:
                continue
            
            if class_name not in predictions:
                predictions[class_name] = {'confidence': 0}
            
            if predictions[class_name]['confidence'] <= confidence:
                predictions[class_name]['confidence'] = confidence
                predictions[class_name]['attributes'] = attributes_class
                predictions[class_name]['index'] = i 

        return predictions
    
    def search(self, filename):
        image_bytes, width, height = self.read_image(filename, self.image_size)
        np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks = self.get_predictions(image_bytes, width, height)
        predictions = self.post_process_predictions(np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks)

        gender, upperbody_query_words, lowerbody_query_words, other_query_words, upperbody_index, lowerbody_index = process_tags(filename, predictions)
        
        search_tags = []
        for li in [upperbody_query_words, lowerbody_query_words, other_query_words]:
            tags = [gender] if gender else []
            for tag in li:
                tag = tag.replace('(', '')
                tag = tag.replace(')', '')
                tag = tag.replace('-', '')
                tags += tag.split(' ')
            search_tags.append(list(set(tags)))
        mask = np_masks[upperbody_index]
        dominant_colors = get_dominant_color(filename, mask)
        st = search_tags[0] + dominant_colors
        st = clean_attributes(st)
        
        items = match_items(self.catalog_data, st)
        response = [[i['pid'],i['url'],i['image_url']] for i in items]
        return response
    
if __name__ == '__main__':
    model = Model()
    model.search_query('test.jpg')