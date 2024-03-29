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

import pika
import uuid


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

def do_work( emb_array, result_dict_predictions, result_dict_class_names, classifier_filename_exp, subset_model_and_class_names):
                        # gw: todo, find a way to append matrices to the right of existing ones, 

    # Classify images
    subset_model, subset_class_names = subset_model_and_class_names
    begin = time.perf_counter()
    subset_predictions = subset_model.predict_proba(emb_array)
    end = time.perf_counter()

    print('classified subset {} images in {} seconds'.format(len(emb_array), end-begin))

    result_dict_predictions[classifier_filename_exp] = subset_predictions
    result_dict_class_names[classifier_filename_exp] = subset_class_names[:]


def str_eq(a,b):
    return str(a) == str(b)

arr_eq = np.frompyfunc(str_eq, 2,1)

# def get_arr_val(arr, idx):
#     return arr[idx]

# arr_get_arr_val = np.frompyfunc(get_arr_val, 2, 1)



def classifier_init(args):

    # init
    # load subset classifiers
    classifier_filename_exp_list = [os.path.expanduser(fname) for fname in args['classifier_filename']]
    subset_model_and_class_names_list = []
    for classifier_filename_exp in classifier_filename_exp_list:
        start = time.perf_counter()
        with open(classifier_filename_exp, 'rb') as infile:
            (subset_model, subset_class_names, _) = pickle.load(infile)
            subset_model_and_class_names_list.append((subset_model, subset_class_names))
        end = time.perf_counter()
        
        print('Loaded classifier subset_model from file {} in {} seconds'.format(classifier_filename_exp, end - start))

    #predictions = None
    #combined_class_names = None
    mgr = multiprocessing.Manager()
    result_dict_predictions = mgr.dict()
    result_dict_class_names = mgr.dict()

    # load msid to human-readable name mapping
    msid_to_name_dict = ms1m_name_list_reader("/home/gaopeng/workspace/ms1m-aligned-full/Top1M_MidList.Name.tsv")


    # Load the subset_model
    print('Loading feature extraction subset_model')
    facenet.load_model(args['model'])
    
    # gw: bookmark 1129 afternoon
    # ----done init----

    # graph = tf.Graph() # gw not working, likely need to enter the context. Anyway, we can use default graph implicitly: https://stackoverflow.com/questions/39614938/why-do-we-need-tensorflow-tf-graph
    # not good practice but reduce our complexity for now
    with tf.Session() as sess:
        def classify_fn(dirname):
            # nonlocal graph

            #nonlocal predictions
            #nonlocal combined_class_names

            # need to be per call basis, not per init basis
            predictions = None
            combined_class_names = None
            

            # predictions = None
            # combined_class_names = None
            # mgr = multiprocessing.Manager()
            # result_dict_predictions = mgr.dict()
            # result_dict_class_names = mgr.dict()



            np.random.seed(seed=args['seed'])

            # input loop for classification

                # e.g.: /home/gaopeng/workspace/ms1m-aligned-full/gw_celeb_3500_20_val/
            # data_dir = input("Input a data dir for classification: ")
            data_dir = dirname
            print("data dir for classification is {}".format(data_dir))
            dataset = facenet.get_dataset(data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            


            paths, label_numbers, label_names = facenet.get_image_paths_and_labelnames(dataset)

            # label_names = [ name.replace('_', ' ') for name in label_names]


            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # gw: tried to move to init
            # Load the subset_model
            #print('Loading feature extraction subset_model')
            #facenet.load_model(args['model'])

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args['batch_size']))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args['batch_size']
                end_index = min((i+1)*args['batch_size'], nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args['image_size'])
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            #------------- classifiying
            print('Testing classifier')




            # done: extracted out one-time expensive task out as initialization (model loading)
            # todo: how to make it a long running process?
            jobs = [multiprocessing.Process(target=do_work, args=(emb_array, result_dict_predictions, result_dict_class_names, classifier_filename_exp_list[i], subset_model_and_class_names_list[i])) for i in range(8)]

            for job in jobs:
                job.start()

            for job in jobs:
                job.join()

            #bp()
            for classifier_filename_exp in classifier_filename_exp_list:
                subset_prediction = result_dict_predictions[classifier_filename_exp]
                subset_class_names = result_dict_class_names[classifier_filename_exp]

                # gw: normalize within each subset, so that later when get Top N labels, we get the top one among each subset
                # dont use it unless you have got a 2nd comparer, e.g. siamsenet
                # subset_prediction = subset_prediction / np.expand_dims(subset_prediction.max(axis = 1), axis=1)
                
                if predictions is None:
                    predictions = subset_prediction
                    combined_class_names = subset_class_names[:]
                    #bp()

                else:
                    predictions = np.concatenate((predictions, subset_prediction), axis=1)
                    combined_class_names = np.concatenate((combined_class_names, subset_class_names[:]), axis=0)

            # bp()


            #N = 8 * 3
            N = 8 * 3
            top_N_class_label_indices = np.argsort(-predictions, axis=1)[:, :N]  # 8 * 3 per subset
            # b_top3_per_row = b[
            #             np.repeat(
            #                             np.expand_dims(
            #                                                 np.array([0,1,2]),
            #                                                 axis=1
            #                             ),
            #                             3,
            #                             axis = 1
            #             )
            #             , columns
            # ]
            #bp()
            top_N_class_label_prob = predictions[
                np.repeat(
                    np.expand_dims(
                        np.arange(top_N_class_label_indices.shape[0]),
                        axis = 1
                    )
                    , N
                    , axis = 1
                )
                , top_N_class_label_indices
            ]
            #bp()
            # msid's
            top_N_class_label_names = np.chararray((top_N_class_label_indices.shape[0], top_N_class_label_indices.shape[1]))
            for i in range(top_N_class_label_names.shape[0]):
                for j in range(top_N_class_label_names.shape[1]):
                    try:
                        top_N_class_label_names[i][j] = combined_class_names[top_N_class_label_indices[i][j]]
                    except:
                        print("i: {}, j:{}, top_N_class_label_indices[i][j]: {}".format(i,j, top_N_class_label_indices[i][j]))
                        raise
            top_N_class_face_dist = np.zeros((top_N_class_label_indices.shape[0], top_N_class_label_indices.shape[1]))
            for i in range(top_N_class_face_dist.shape[0]):
                msid_to_img_fpath_dict = {}
                #bp()
                for j in range(top_N_class_face_dist.shape[1]):
                    msid = combined_class_names[top_N_class_label_indices[i][j]]
                    img_fname = get_random_img_under_msid(msid)  # TODO
                    msid_to_img_fpath_dict[msid] = os.path.join(msid, img_fname)
                    
                test_img_fname = os.path.basename(paths[i])
                test_img_dirname_person_folder = os.path.basename(os.path.dirname(paths[i]))
                test_img_dirname_tmp_folder = os.path.basename(os.path.dirname(os.path.dirname(paths[i])))
                # second_compare_rpc
                
                msid_to_face_dist_dict = pickle.loads(
                    second_compare_rpc.call(
                        pickle.dumps((msid_to_img_fpath_dict, os.path.join(test_img_dirname_tmp_folder, test_img_dirname_person_folder, test_img_fname)))))

                
                for j in range(top_N_class_face_dist.shape[1]):
                    msid = combined_class_names[top_N_class_label_indices[i][j]]
                    top_N_class_face_dist[i][j] = msid_to_face_dist_dict[msid]
                
                    
            #bp()
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_names = np.array([combined_class_names[idx] for idx in best_class_indices])
            #best_class_names = np.arang;  # gw: todo: use ix_ to make a ndarray with row being test_indices, and col (only 1) being the best class name
            # best_class_indices = np.argsort(predictions, axis=1)[-8:] #gw: todo, calcuate embedding distance again amont 8 rfc results
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            def show_human_name_or_raw_class_name(msid):
                if msid in msid_to_name_dict:
                    for name in msid_to_name_dict[msid]:
                        if "@en" in name:
                            return name
                        elif "@zh" in name:
                            return name
                        elif "@zh-Hant" in name:
                            return name

                return msid

            result_str = ''
            #result = []
            result = {} # gw: use dict foe easier indexing in client
            for i in range(len(best_class_indices)):
                line = 'actual name (msid): {} ({}),\t, pred name (msid): {} ({}), \t prob:  {:.3f}, \t img: {}'.format (show_human_name_or_raw_class_name(label_names[i]),
                                                                                               label_names[i],
                                                                                                show_human_name_or_raw_class_name(combined_class_names[best_class_indices[i]]),
                                                                                                                 combined_class_names[best_class_indices[i]],
                                                                                           best_class_probabilities[i],
                                                                                           paths[i])
                result_str += (line + '\n')
                print(line)
                try: 
                    single_prediction_dict = {
                        'screenId' : int(label_names[i]),
                        'best': {
                            'name': show_human_name_or_raw_class_name(combined_class_names[best_class_indices[i]]).split('@')[0].strip('"'),
                            'prob': best_class_probabilities[i],
                        },
                        'topN': [
                            {
                                'name': show_human_name_or_raw_class_name(combined_class_names[top_N_class_label_indices[i][j]]).split('@')[0].strip('"'),
                                'prob': top_N_class_label_prob[i][j],
                                'dist': top_N_class_face_dist[i][j],
                            } for j in range(N)
                        ]
                    }
                    result[single_prediction_dict['screenId']] = single_prediction_dict
                except:
                    single_prediction_dict = {
                        'screenId': None,
                        'name': None,
                        'prob': None,
                        'best': None,
                        'topN': []
                    }

                # label_name will be person screen id: e.g. 1,2,3
                #result.append({label_names[i]: single_prediction_dict})  # label_name[i] is folder name, also the key for uploading, e.g. "person-1": person-1.jpg
                


            # accuracy = np.mean(np.equal(best_class_indices, label_numbers))
            # accuracy = np.mean(np.equal(best_class_indices, label_names))
            # bp()
            accuracy = np.mean(arr_eq(best_class_names, label_names))
            accuracy_str = 'Accuracy: {:.13f}'.format(accuracy)
            print(accuracy_str)
            result_str += (accuracy_str + '\n')
            #return result_str

            #bp()

            return json.dumps(result)

    return classify_fn

def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

# python src/classifier_multi.py --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..9}_rfc_classifier.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 --use_split_dataset
# def parse_arguments(argv):
def parse_arguments():
    args = {}
    args['model'] = '/home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb'
    args['classifier_filename'] = ['~/models/gw_subset_{}_svc_classifier.pkl'.format(i) for i in range(1,9)]
    args['batch_size'] = 1536
    args['image_size'] = 160
    args['seed'] = 666
    args['min_nrof_images_per_class'] = 20
    args['nrof_train_images_per_class'] = 15
    

    return args






# ----
# from celery import Celery
# from celery.signals import worker_init

# classify_delegate=None

# app = Celery('tasks', broker='pyamqp://guest@localhost//')

# @worker_init.connect
# def task_init(**kwargs):
    
#     # seting daemon to false, otherwise cant use multiprocessing
#     # https://github.com/celery/celery/issues/1709#issuecomment-324802431
#     print("init config, before: {}".format(multiprocessing.current_process()._config))
#     multiprocessing.current_process()._config['daemon'] = False
#     print("init config, after: {}".format(multiprocessing.current_process()._config))
    
#     #bp()
#     tmp_args = parse_arguments()
#     classify_fn = classifier_init(tmp_args)  # now the global val classify_delegate holds the actual classifying function using tensorflow
#     global classify_delegate
#     def classify_delegate(dname):
#         return classify_fn(dname)

    


# @app.task
# def classify_directory(dname):
#     print("task config, before: {}".format(multiprocessing.current_process()._config))
#     multiprocessing.current_process()._config['daemon'] = False
#     print("task config, after: {}".format(multiprocessing.current_process()._config))

    
#     global classify_delegate
#     return classify_delegate(dname)

# gw: moved to celery worker_init
# if __name__ == '__main__':
#    init(parse_arguments())
    # sample invocation:
    # (venv_facenet) gaopeng@gw-cm6870:~/workspace/python3.6/facenet$ python src/classifier_daemon_subsets_with_multiproc.py --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..8}_svc_classifier.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 



# gw: first compare rpc server
connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))

channel = connection.channel()

channel.queue_declare(queue='rpc_queue')

# gw: reduce prefetch count to fix "connection reset by peer" error
# https://stackoverflow.com/questions/35438843/rabbitmq-error-timeout
# channel.basic_qos(prefetch_count=10)

def on_request(ch, method, props, body):
    #n = int(body)

    #print(" [.] fib(%s)" % n)
    #response = fib(n)


    # dirname = str(body)

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


# seconcd compare rpc client
class SecondCompareRpcClient(object):
    def __init__(self):


        credentials = pika.PlainCredentials('starface', 'f4c3st4r')
        parameters = pika.ConnectionParameters('192.168.0.104',
                                               5672,
                                               '/',
                                               credentials)
        
        self.connection = pika.BlockingConnection(parameters)

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(self.on_response, no_ack=True,
                                   queue=self.callback_queue)

    def on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    def call(self, body):
        self.response = None
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key='second_comparison_queue_7',
                                   properties=pika.BasicProperties(
                                         reply_to = self.callback_queue,
                                         correlation_id = self.corr_id,
                                         ),
                                   body=body)
        while self.response is None:
            self.connection.process_data_events()
        return self.response

second_compare_rpc = SecondCompareRpcClient()



if __name__ == '__main__':


    print("task config, before: {}".format(multiprocessing.current_process()._config))

    tmp_args = parse_arguments()
    classify_fn = classifier_init(tmp_args)  # now the global val classify_delegate holds the actual classifying function using tensorflow
    global classify_delegate
    def classify_delegate(dname):
        return classify_fn(dname)

    
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_request, queue='rpc_queue')

    print(" [x] Awaiting RPC requests")
    channel.start_consuming()
    
