from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pprint

import sys
import os
import logging

sys.path.append('../..')
sys.path.append('../efficientnet')
sys.path.append(os.path.join(os.path.dirname(__file__), 'FaceLib'))
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
from google.cloud import bigquery, storage
import argparse

import webcolors
import matplotlib.colors as mc
from facelib import FaceDetector, AgeGenderEstimator

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

def get_gender(image_path):
    face_detector = FaceDetector()
    age_gender_detector = AgeGenderEstimator()

    image = cv2.imread(image_path)
    faces, boxes, scores, landmarks = face_detector.detect_align(image)
    if faces.tolist():
        genders, ages = age_gender_detector.detect(faces)
    else:
        genders = [None]
    return genders[0]
         
class Model:
    min_score_threshold = 0.5
    image_size = 640
    label_map_dict = None
    attribute_index = None
    image_input = None
    predictions = None
    sess = None
    catalog_data = None
    table_client = None
    table_catalog_name = None
    table_inference_name = None
    
    def __init__(self,
                 model_name='attribute_mask_rcnn',
                 path_to_checkpoint='projects/fashionpedia/checkpoints/fashionpedia-spinenet-143/model.ckpt',
                 path_to_config='projects/fashionpedia/configs/yaml/spinenet143_amrcnn.yaml',
                 path_to_label_map='projects/fashionpedia/dataset/fashionpedia_label_map.csv',
                 path_to_attribute_map='projects/fashionpedia/dataset/attributes_label_map.json',
                 table_catalog_name='`maximal-furnace-783.video_commerce_experiments.fk_catalog_data`',
                 table_inference_name='`maximal-furnace-783.video_commerce_experiments.fk_catalog_inferred_data`',
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
        
        self.table_client = bigquery.Client()
        self.table_catalog_name = table_catalog_name
        self.table_inference_name = table_inference_name

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
        max_index, max_score = 0, 0
        for i in range(len(np_scores)):
            if np_classes[i] not in range(12) and np_classes[i] not in range(20,22):
                continue
            if np_scores[i] > max_score:
                max_score = np_scores[i]
                max_index = i
        
        class_id = np_classes[max_index]
        class_name = self.label_map_dict[class_id]['name']
        confidence = np_scores[max_index]
        
        attribute_list = np_attributes[max_index]
        attributes = [self.attribute_index[j] for j in range(len(attribute_list)) if attribute_list[j] > self.min_score_threshold]
        attribute_names = [self.attribute_index[j]['name'] for j in range(len(attribute_list)) if attribute_list[j] > self.min_score_threshold]
        attribute_scores = [attribute_list[j] for j in range(len(attribute_list)) if attribute_list[j] > self.min_score_threshold]

        return max_index, class_name, attribute_names
    
    def search_query(self, filename):
        image_bytes, width, height = self.read_image(filename, self.image_size)
        np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks = self.get_predictions(image_bytes, width, height)
        max_index, class_name, attribute_names = self.post_process_predictions(np_boxes, np_scores, np_classes, np_attributes, np_masks, encoded_masks)
        img = Image.open(filename)
        mask = np_masks[max_index]
        dominant_colors = get_dominant_color(img, mask)
        gender = get_gender(filename)
#         print(filename)
#         print(class_name, gender, attribute_names, dominant_colors)
#         print()
        return self.search(class_name, gender, attribute_names, dominant_colors)
    
    def search(self, class_name, gender, attribute_names, dominant_colors):
        if gender == 'Male':
            gender = 'man'
        elif gender == 'Female':
            gender = 'woman'
        else:
            gender = '%man'
        gender_condition = ' AND gender LIKE "' + gender + '"'
        
        attribute_condition = ''
        for a in attribute_names:
            attribute_condition += ' AND "' + a + '" IN UNNEST(attributes)'
         
        color_condition = ' AND "' + dominant_colors[0] + '" IN UNNEST(product_tags)'
        category_condition = ' product_category = "' + class_name + '"'
        
        queryI = 'SELECT product_id FROM ' + self.table_inference_name + ' WHERE' + category_condition + attribute_condition
        queryP = 'SELECT * FROM ' + self.table_catalog_name + ' WHERE product_id IN (' + queryI + ')' + gender_condition + color_condition + ' LIMIT 5'
                
        rows = self.table_client.query(queryP)
        while not rows.done():
            pass
        
        response = []
        for r in rows:
            image_url = r['image_url'].replace('{@height}','640').replace('{@width}','640').replace('?q={@quality}', '')
            responseItem = [r['product_id'], r['page_url'], image_url]
            response.append(responseItem)
#             print(responseItem)
#             print()
        return response
    
if __name__ == '__main__':
    model = Model()
    print(model.search_query('test.jpg'))