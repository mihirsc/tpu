import os
from fp import Model
import time
import logging
from PIL import Image
from google.cloud import storage
from flask_cors import CORS, cross_origin
from flask import Flask, request, send_file, jsonify
from time import perf_counter

def upload_to_bucket(bucket_prefix, dir_name, filename, bucket):
    blob = bucket.blob(bucket_prefix + filename)
    blob.upload_from_filename(os.path.join(dir_name, filename))

logger=logging.getLogger('logger')  
logger.setLevel(logging.DEBUG)

model = Model()

bucket_client = storage.Client()
bucket_name = 'ds-camera-video-ai'
bucket_prefix = 'video-commerce/datasets/v0_system_query_images/'
bucket = bucket_client.get_bucket(bucket_name)

app = Flask(__name__)
cors = CORS(app)
app.config["DEBUG"] = True
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/single_image_inference', methods=['POST'])
@cross_origin()
def single_image_inference():
    x = perf_counter()
    image = request.files['query']
    img = Image.open(image)
    filename = str(int(time.time()))+ "_" + image.filename
    img.save(os.path.join('query_images', filename))
    upload_to_bucket(bucket_prefix, 'query_images', filename, bucket)
    items = model.search_query(os.path.join('query_images', filename))
    print('Search took',perf_counter()-x,'seconds')
    print()
    return jsonify(results=items)

@app.route('/multi_image_inference', methods=['POST'])
@cross_origin()
def multi_image_inference():
    x = perf_counter()
    images = request.files.getlist('queries')
    items_all = []
    for image in images:
        img = Image.open(image)
        filename = str(int(time.time()))+ "_" + image.filename
        img.save(os.path.join('query_images', filename))
        upload_to_bucket(bucket_prefix, 'query_images', filename, bucket)
        items = model.search_query(os.path.join('query_images', filename))
        items_all.append(items)
    print('Search took',perf_counter()-x,'seconds')
    print()
    return jsonify(results=items_all)

app.run(host='0.0.0.0')