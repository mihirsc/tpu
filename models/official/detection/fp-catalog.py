from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

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
from time import perf_counter
import argparse

from map_fashionpedia_to_flipkart import map_fashionpedia2flipkart as process_tags

import webcolors
import matplotlib.colors as mc

def closest_colour(requested_colour):
    min_colours = {}
    for name, key in mc.CSS4_COLORS.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_dominant_color(img, mask):
    common_colors = {}
    common_colors['red'] = ['coral', 'crimson', 'darkred', 'darksalmon', 'firebrick', 'indianred', 'lightcoral', 'lightsalmon', 'maroon', 'mistyrose', 'red', 'salmon', 'sienna', 'tomato']
    common_colors['blue'] = ['aliceblue', 'aqua', 'aquamarine', 'azure', 'blue', 'blueviolet', 'cadetblue', 'cornflowerblue', 'cyan', 'darkblue', 'darkcyan', 'darkslateblue', 'darkturquoise', 'deepskyblue', 'dodgerblue', 'lightblue', 'lightcyan', 'lightskyblue', 'lightsteelblue', 'mediumaquamarine', 'mediumblue', 'mediumslateblue', 'mediumturquoise', 'midnightblue', 'navy', 'paleturquoise', 'powderblue', 'royalblue', 'skyblue', 'slateblue', 'steelblue', 'teal', 'turquoise']
    common_colors['yellow'] = ['cornsilk', 'darkgoldenrod', 'gold', 'goldenrod', 'lemonchiffon', 'lightgoldenrodyellow', 'lightyellow', 'palegoldenrod', 'yellow']
    common_colors['green'] = ['chartreuse', 'darkgreen', 'darkkhaki', 'darkolivegreen', 'darkseagreen', 'forestgreen', 'green', 'greenyellow', 'honeydew', 'khaki', 'lawngreen', 'lightgreen', 'lightseagreen', 'lime', 'limegreen', 'mediumseagreen', 'mediumspringgreen', 'mintcream', 'olive', 'olivedrab', 'palegreen', 'seagreen', 'springgreen', 'yellowgreen']
    common_colors['black'] = ['black']
    common_colors['white'] = ['antiquewhite', 'floralwhite', 'ghostwhite', 'ivory', 'navajowhite', 'snow', 'white', 'whitesmoke']
    common_colors['grey'] = ['darkgray', 'darkgrey', 'darkslategray', 'darkslategrey', 'dimgray', 'dimgrey', 'gainsboro', 'gray', 'grey', 'lightgray', 'lightgrey', 'lightslategray', 'lightslategrey', 'slategray', 'slategrey']
    common_colors['brown'] = ['brown', 'burlywood', 'chocolate', 'peru', 'rosybrown', 'saddlebrown', 'sandybrown']
    common_colors['cream'] = ['beige', 'bisque', 'blanchedalmond', 'linen', 'oldlace', 'moccasin', 'papayawhip', 'peachpuff', 'seashell', 'tan', 'wheat']
    common_colors['purple'] = ['darkmagenta', 'darkorchid', 'darkviolet', 'fuchsia', 'indigo', 'lavender', 'lavenderblush', 'magenta', 'mediumorchid', 'mediumpurple', 'mediumvioletred', 'orchid', 'palevioletred', 'plum', 'purple', 'rebeccapurple', 'thistle', 'violet'] 
    common_colors['orange'] = ['darkorange', 'orange', 'orangered']
    common_colors['pink'] = ['deeppink', 'hotpink', 'lightpink', 'pink']
    common_colors['silver'] = ['silver']
    ni = np.array(img)
    counter = {}
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if mask[i][j]:
                r, g, b = ni[i][j]
                r = r - (r%20)
                g = g - (g%20)
                b = b - (b%20)
                c = (r,g,b)
                if c not in counter:
                    counter[c] = 0
                counter[c] += 1
    colors = [[c,counter[c]] for c in counter]
    colors.sort(key=lambda x:x[1], reverse=True)
    d_c = list(set([closest_colour(c[0]) for c in colors[:2]]))
    dominant_colors = []
    for c in d_c:
        check = True
        for common_color in common_colors:
            if c in common_colors[common_color]:
                dominant_colors.append(common_color)
                check = False
        if check:
            dominant_colors.append(c)
    return list(set(dominant_colors))
         
class Model:
    min_score_threshold = 0.5
    image_size = 640
    label_map_dict = None
    attribute_index = None
    image_input = None
    predictions = None
    sess = None
    catalog_data = None
    
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

        tf.compat.v1.disable_eager_execution() #For TF2
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
    
    def search_query(self, filename):
        logger.info('Filename: ' + filename)
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

        img = Image.open(filename)
        mask = np_masks[upperbody_index]
        dominant_colors = get_dominant_color(img, mask)
        st = search_tags[0] + dominant_colors
        st = [s.lower() for s in st]
        logger.info('Query Tags: ' + ' '.join([s for s in st]))
        
        items = match_items(self.catalog_data, st)
        for i in items:
            logger.info('Product Tags: ' + ' '.join([t for t in i['tags']]))
            logger.info('Scores: ' + str(i['scores']))
        response = [[i['pid'],i['url'],i['image_url']] for i in items]

        return response