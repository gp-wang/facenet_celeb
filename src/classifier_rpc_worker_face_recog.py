"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import traceback
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
import time
import multiprocessing
from pdb import set_trace as bp
import functools
import json
import random
from functools import reduce
import pika
import uuid

import numpy as np
from collections import namedtuple


from sqlalchemy import and_, or_
from sqlalchemy import Column
from sqlalchemy import create_engine
from sqlalchemy import DateTime
from sqlalchemy import Float
from sqlalchemy import ForeignKey
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import VARCHAR
from sqlalchemy import CHAR
from sqlalchemy import BINARY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.orm import Session

import traceback
from datetime import datetime
# my db model for imdb celebrity, original on c5 imdb_celeb_parser
from model import ImdbCelebrity, ImdbMsid
# import model


# Constants -----------------------------------------------


DB_USER = 'imdb_celeb_parser'
DB_PASSWORD = 'imdb_celeb_parser'
DB_HOST = '192.168.0.105'
DB_PORT = '3306'
DB_NAME = 'imdb_celeb_parser'




Prediction = namedtuple('Prediction', ['msid', 'prob'])

def curate_top_predictions(predictions):
    # predictions: [Prediction]
    assert len(predictions) >= 5, "not enough preds to curate"
    top_5_predictions = predictions[:5]
    bins = range(120, 0, -20)
    bins_with_offset_decimal = list(map(lambda x: (x-10) / 100, bins))
    prediction_probs = [pred.prob for pred in top_5_predictions]
    predictions_bin_indices = np.digitize(prediction_probs, bins_with_offset_decimal)  # indice are 1-based
    first_bin = min(predictions_bin_indices)
    curated_predictions = sorted([pred for idx,pred in enumerate(top_5_predictions) if predictions_bin_indices[idx] == first_bin ], key=lambda p: p.prob, reverse=True)
    return curated_predictions
    

def format_list_of_float(l_of_float, format_str):
    return list(map(lambda f: format_str.format(f), l_of_float))

BASE_TRAIN_SET_PATH = '/home/gaopeng/workspace/ms1m-aligned-full/filtered_by_imdb_top_4000_celeb/train'

def get_random_img_under_msid(msid):
    full_dirname = os.path.join(BASE_TRAIN_SET_PATH, msid)
    imgs = [fname  for fname in os.listdir(full_dirname) if fname.endswith(".jpg")]
    
    #return random.choices(imgs, k=5)
    return random.choice(imgs)
    
def ms1m_name_list_reader(fname):
    assert os.path.isfile(fname), "{} doesn't exist".format(fname)

    result_dict = {}
    with open(fname, "r") as fin:
        for line in fin.readlines():
            msid, name = line.strip().split("\t")
            if not msid in result_dict:
                result_dict[msid] = [name]
            else:
                result_dict[msid].append(name)

    return result_dict

NOF_NEIGHBORS = 5

# gw: copied and modifed from facenet.py
def get_dataset(path, has_class_directories=True):
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
                    if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        image_paths = get_image_paths(facedir)
        dataset.append((class_name, image_paths))
    return dataset

# gw: copied and modifed from facenet.py
def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths


def classifier_init():

    # init
    # load subset classifiers
    # clf_fpath = '/home/gaopeng/models/gw_ms1m_knn_model_12302018.clf'  # gw: cpu=1, knn=2

    # gw: working model b4 0105, 4k celeb
    # clf_fpath = '/home/gaopeng/models/gw_ms1m_knn_nof_5_model_x8_12312018.clf'  # gw: cpu=8, knn=5
    # label_fpath = '/home/gaopeng/models/ms1m_labels'

    # gw: model 0105, 80k ms1m (full)
    # clf_fpath = '/home/gaopeng/workspace/python3.6/face_recog/code_base/face_recognition/examples/trained_knn_model'  # gw: cpu=8, knn=5
    # label_fpath = '/home/gaopeng/workspace/python3.6/face_recog/code_base/face_recognition/examples/msid_label_list'

    # latest
    clf_fpath = '/home/gaopeng/workspace/python3.6/face_recog/code_base/face_recognition/examples/trained_knn_model'
    label_fpath = '/home/gaopeng/workspace/python3.6/face_recog/code_base/face_recognition/examples/msid_label_list'
    
    with open(clf_fpath, 'rb') as f:
        clf = pickle.load(f)

    with open(label_fpath, 'rb') as f:
        y = pickle.load(f)

        
    def predict_one_img(img_fpath, y=None):
        # bp()
        return predict(img_fpath, knn_clf=clf, distance_threshold=0.6, y=y)

    msid_to_name_dict = ms1m_name_list_reader("/home/gaopeng/workspace/ms1m-aligned-full/Top1M_MidList.Name.gw.tsv")

    def show_human_name_or_raw_class_name(msid):
        human_name = None

        def remove_at(name):
            if '@' in name:
                return name.split('@')[0].strip('"')
            else:
                return name

        if msid in msid_to_name_dict:
            for name in msid_to_name_dict[msid]:
                if "@en" in name:
                    return remove_at(name)
                    
                elif "@zh" in name:
                    return remove_at(name)

                elif "@zh-Hant" in name:
                    return remove_at(name)

        else:
            return msid



    #  DB init
    db_engine = create_engine("mysql+mysqldb://{user}:{pwd}@{host}:{port}/{dbname}".format(
        user=DB_USER,
        pwd=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME
    ))
    db_session = Session(db_engine)
    imdb_celeb_query = db_session.query(ImdbCelebrity, ImdbMsid)
    # TODO: Join


    


    # given a dir:
    # dir\
    #     person1\
    #         img1.jpg
    #         img2.jpg
    #     person2\
    #         img1.jpg
    # return the classification result
    def predict_dir(dirname):

        #bp()
        dataset = get_dataset(dirname)  # tuple of (class_name, img_paths)
        result = {}
        
        for (label_name, img_paths) in dataset:
            label = int(label_name)

            single_prediction_dict = {
                'screenId': None,
                'best': None,
                'topN': []
            }
            
            try:
                predictions = predict_one_img(img_paths[0], y=y)
                if predictions:

                    top_pred = predictions[0]
                    msid = top_pred.msid

                    bio, avartar, birthYear, deathYear, professions, knownForTitles = None, None, None, None, None, None
                    primaryName = "unknown"
                    tb = None
                    try:
                        imdb_celeb, imdb_msid = imdb_celeb_query.filter(ImdbMsid.msid == msid).filter(ImdbMsid.nconst == ImdbCelebrity.nconst).all()[0]

                        primaryName = imdb_celeb.primaryName
                        bio = imdb_celeb.bio
                        avartar = imdb_celeb.avartar_blob
                        birthYear = imdb_celeb.birthYear
                        deathYear = imdb_celeb.deathYear
                        professions = imdb_celeb.primaryProfession
                        knownForTitles = imdb_celeb.knownForTitles
                        
                    except IndexError as ie:
                        print("db lookup gives empty rows, err: %r " % ie)
                        tb = traceback.format_exc()
                    except Exception as e:
                        print("db lookup unkown exception: %r " % e)
                        tb = traceback.format_exc()
                    else:
                        tb = ""

                    finally:
                        if tb:
                            print(tb)

                    # gw: deprecated, use imdb_celeb instead
                    # predicted_human_name = show_human_name_or_raw_class_name(top_pred.msid)

                    single_prediction_dict = {
                        'screenId' : label,
                        'best': {
                            # img_paths[0]: assume there is only one image for each person folder
                            # 'name': predicted_human_name,
                            'name': primaryName,
                            # put as 0.0 dummy value since we are using knn (prob are for earlier svc model)
                            'prob': top_pred.prob,

                            # gw: new field since 02022019
                            'bio': bio,
                            'avartar' : avartar,
                            'birthYear': birthYear ,
                            'deathYear': deathYear ,
                            'professions': professions ,
                            'knownForTitles': knownForTitles ,
                        },
                        'topN': [
                            {
                                'name': show_human_name_or_raw_class_name(pred.msid),
                                'prob': pred.prob
                            } for pred in predictions
                        ]
                    }
                else:
                    pass  # use init value of single_prediction_dict
                    
            except Exception as e:
                print("failed for dirname: {}, exception: {}".format(dirname, e))
                tb = traceback.format_exc()
                pass            # use init value of single_prediction_dict
            else:
                tb = ""
            finally:
                if tb:
                    print(tb)
                    
            result[label] = single_prediction_dict
        return json.dumps(result)



    return predict_dir


# gw: first compare rpc server
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='rpc_queue')

def on_request(ch, method, props, body):
    # gw: from bytes to string, properway is to decode:
    # https://stackoverflow.com/questions/606191/convert-bytes-to-a-string
    dirname=body.decode('utf-8')

    global classify_delegate
    response = classify_delegate(dirname)

    ch.basic_publish(exchange='',
                     routing_key=props.reply_to,
                     properties=pika.BasicProperties(correlation_id = \
                                                         props.correlation_id),
                     body=str(response))
    ch.basic_ack(delivery_tag = method.delivery_tag)




import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6, y=None):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param X_img_path: path to image to be recognized
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
p    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.

    ;param y: array to map index of class to its label string (gw: its shape should around 4k)
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.


    gw: notes and learnings:
    There are two concepts to distinguish:
    1. population matrix: this consists of all training data points, in my case, it is 380k in size (380k face images belonging to 4k classes)
    2. class label array: this consisits of all class labels, in my case, 4k classes

    There are two methods to distinguish:
    1. knn.closest_neighbors: the returned indices are population matrix
    2. knn.predict_proba: the returned indices are class label array

    Don't ever mix them up
    
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_img = face_recognition.load_image_file(X_img_path)
    # X_face_locations = face_recognition.face_locations(X_img)
    # [print("{}".format(loc)) for loc in X_face_locations  ]
    # X_face_locations = face_recognition.face_locations(X_img, number_of_times_to_upsample=2, model='cnn')  # gw
    # X_face_locations = face_recognition.face_locations(X_img, model='cnn')  # gw
    # gw; using cropped image (no alignment)

    # gw: top, right, bottom, left
    # gw: y0, x1, y1, x0
    # bp()
    X_face_locations = [(0,X_img.shape[0], X_img.shape[1],0)]

    # If no faces are found in the image, return an empty result.
    # if len(X_face_locations) == 0:
    #    return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    # closest_neighbors = knn_clf.kneighbors(faces_encodings, n_neighbors=1,  return_distance=True)
    # closest_neighbors = knn_clf.kneighbors(faces_encodings, n_neighbors=NOF_NEIGHBORS,  return_distance=True)
    #bp()
    # closest_distances = [ dist  for dist in closest_neighbors[0][0]]

    # bp()
    closest_probs_for_faces = knn_clf.predict_proba(faces_encodings)  # plural because may have multiple faces in faces_encodings
    closest_probs_for_one_face = closest_probs_for_faces[0]                         # we expect only one face per call
    top_5_indices = closest_probs_for_one_face.argsort()[-5:].tolist()
    top_5_prob = list(map(lambda idx: closest_probs_for_one_face[idx], top_5_indices))
    top_5_label = list(map(lambda idx: knn_clf.classes_[idx], top_5_indices))

    print("before curation: top 5 labels {}".format(top_5_label))

    top_5_predictions = [Prediction(msid=msid, prob=prob) for prob, msid in zip(top_5_prob, top_5_label)]

    curated_predictions = curate_top_predictions(top_5_predictions)

    print("after curation; , prob: {}, label: {}, std: {}".format(
        # [item.label_index for item in curated_predictions],
        format_list_of_float( [item.prob for item in curated_predictions] , "{0:.5f}"),
        [item.msid for item in curated_predictions],
        np.std([item.prob for item in curated_predictions])))

    # bp()

    # -- use avg_dist
    # avg_dist = reduce(lambda x, y: x + y, closest_distances) / len(closest_distances)
    # are_matches = [closest_neighbors[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    # gw: use avg dist as criteria
    # are_matches = [ True if avg_dist <= distance_threshold else False ]   # list of size 1 because 1 face per img
    
    # -- use bin to determine match
    is_match = False
    if not curated_predictions or len(curated_predictions) == 0:
        is_match = False
    else:
        if curated_predictions[0].prob < 0.25:  # 0.25 = 0.2 + offset 0.05 margin
            is_match = False
        else:
            is_match = True
    

    # print("k-nbrs: {} {}".format(closest_neighbors[0].tolist()[0], closest_neighbors[1].tolist()[0]))
    # k_nbr_labels = list(map(lambda idx: y[idx], closest_neighbors[1].tolist()[0]))
    # print("k-nbrs-labels: {}".format(k_nbr_labels))

    # Predict classes and remove classifications that aren't within the threshold
    # return [(pred, avg_dist) if rec else ("unknown", 1.0) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


    if is_match:
        return curated_predictions
    else:
        return []








# usage:
# 
# python src/classifier_multi.py --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..9}_rfc_classifier.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 --use_split_dataset
# def parse_arguments(argv):

if __name__ == '__main__':


    print("task config, before: {}".format(multiprocessing.current_process()._config))


    predict_dir = classifier_init()  # now the global val classify_delegate holds the actual classifying function using tensorflow
    global classify_delegate
    def classify_delegate(dname):
        return predict_dir(dname)

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue='rpc_queue')

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()


    
