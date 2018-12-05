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

    predictions = None
    combined_class_names = None
    mgr = multiprocessing.Manager()
    result_dict_predictions = mgr.dict()
    result_dict_class_names = mgr.dict()

    # load msid to human-readable name mapping
    msid_to_name_dict = ms1m_name_list_reader("/home/gaopeng/workspace/ms1m-aligned-full/Top1M_MidList.Name.tsv")

    # gw: bookmark 1129 afternoon
    # ----done init----

    # graph = tf.Graph() # gw not working, likely need to enter the context. Anyway, we can use default graph implicitly: https://stackoverflow.com/questions/39614938/why-do-we-need-tensorflow-tf-graph
    # not good practice but reduce our complexity for now
    def classify_fn(dirname):
        # nonlocal graph
        
        nonlocal predictions
        nonlocal combined_class_names


    # predictions = None
    # combined_class_names = None
    # mgr = multiprocessing.Manager()
    # result_dict_predictions = mgr.dict()
    # result_dict_class_names = mgr.dict()

        with tf.Session() as sess:

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

            # Load the subset_model
            print('Loading feature extraction subset_model')
            facenet.load_model(args['model'])

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

                if predictions is None:
                    predictions = subset_prediction
                    combined_class_names = subset_class_names[:]
                    #bp()

                else:
                    predictions = np.concatenate((predictions, subset_prediction), axis=1)
                    combined_class_names = np.concatenate((combined_class_names, subset_class_names[:]), axis=0)

            # bp()

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
            for i in range(len(best_class_indices)):
                line = 'actual name (msid): {} ({}),\t, pred name (msid): {} ({}), \t prob:  {:.3f}, \t img: {}'.format (show_human_name_or_raw_class_name(label_names[i]),
                                                                                               label_names[i],
                                                                                           show_human_name_or_raw_class_name(combined_class_names[best_class_indices[i]]),
                                                                                                                 combined_class_names[best_class_indices[i]],
                                                                                           best_class_probabilities[i],
                                                                                           paths[i])
                result_str += (line + '\n')
                print(line)

            # accuracy = np.mean(np.equal(best_class_indices, label_numbers))
            # accuracy = np.mean(np.equal(best_class_indices, label_names))
            # bp()
            accuracy = np.mean(arr_eq(best_class_names, label_names))
            accuracy_str = 'Accuracy: {:.13f}'.format(accuracy)
            print(accuracy_str)
            result_str += (accuracy_str + '\n')
            return result_str

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
from celery import Celery
from celery.signals import worker_init

classify_delegate=None

app = Celery('tasks', broker='pyamqp://guest@localhost//')

@worker_init.connect
def task_init(**kwargs):
    
    # seting daemon to false, otherwise cant use multiprocessing
    # https://github.com/celery/celery/issues/1709#issuecomment-324802431
    print("init config, before: {}".format(multiprocessing.current_process()._config))
    multiprocessing.current_process()._config['daemon'] = False
    print("init config, after: {}".format(multiprocessing.current_process()._config))
    
    #bp()
    tmp_args = parse_arguments()
    classify_fn = classifier_init(tmp_args)  # now the global val classify_delegate holds the actual classifying function using tensorflow
    global classify_delegate
    def classify_delegate(dname):
        return classify_fn(dname)

    


@app.task
def classify_directory(dname):
    print("task config, before: {}".format(multiprocessing.current_process()._config))
    multiprocessing.current_process()._config['daemon'] = False
    print("task config, after: {}".format(multiprocessing.current_process()._config))

    
    global classify_delegate
    return classify_delegate(dname)

# gw: moved to celery worker_init
# if __name__ == '__main__':
#    init(parse_arguments())
    # sample invocation:
    # (venv_facenet) gaopeng@gw-cm6870:~/workspace/python3.6/facenet$ python src/classifier_daemon_subsets_with_multiproc.py --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..8}_svc_classifier.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 

