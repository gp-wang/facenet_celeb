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


from pdb import set_trace as bp

def str_eq(a,b):
    return str(a) == str(b)

arr_eq = np.frompyfunc(str_eq, 2,1)

# def get_arr_val(arr, idx):
#     return arr[idx]

# arr_get_arr_val = np.frompyfunc(get_arr_val, 2, 1)

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, label_numbers, label_names = facenet.get_image_paths_and_labelnames(dataset)

            label_names = [ name.replace('_', ' ') for name in label_names]
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                # gw: TODO: understand this line and apply it for single images
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            classifier_filename_exp_list = [os.path.expanduser(fname) for fname in args.classifier_filename]

            # gw: TRAIN not usable until modified for multi classifiers
            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                #model = SVC(kernel='linear', probability=True)
                model = RFC(n_jobs=8, n_estimators=100)
                #bp()
                model.fit(emb_array, label_numbers)
            
                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names, emb_array), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                print('Testing classifier')
                predictions = None
                combined_class_names = None
                train_emb_array = None
                top_class_indices = None
                M = 30
                N = 8 * M
                i = 0
                for classifier_filename_exp in  classifier_filename_exp_list:
                    # gw: todo, find a way to append matrices to the right of existing ones, 
                    
                    # Classify images
                    
                    with open(classifier_filename_exp, 'rb') as infile:
                        (model, class_names, subset_train_emb_array) = pickle.load(infile)
                        
                    print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                    print("subsuet_train_emb_array: {}".format(subset_train_emb_array.shape))

                    subset_predictions = model.predict_proba(emb_array)
                    # gw: normalize within each subset, so that later when get Top N labels, we get the top one among each subset
                    subset_predictions = subset_predictions / np.expand_dims(subset_predictions.max(axis = 1), axis=1)
                    if predictions is None:

                        predictions = subset_predictions
 
                        combined_class_names = class_names[:]
                        train_emb_array = subset_train_emb_array

                        # top_class_indices = np.argsort(-predictions, axis=1)[:, :M]
                        # top_class_indices += 
                        
 
                    else:
                        predictions = np.concatenate((predictions, subset_predictions), axis=1)
                        combined_class_names = np.concatenate((combined_class_names, class_names[:]), axis=0)
                        #bp()
                        train_emb_array = np.concatenate((train_emb_array, subset_train_emb_array), axis = 0)
                        # top_class_indices = np.concatenate(top_class_indices, np.argsort(-predictions, axis=1)[:, :M])

                    i += 1


                top_N_class_label_indices = np.argsort(-predictions, axis=1)[:, :N]  # 8 * 3 per subset

                #map_to_train_emb_array = np.frompyfunc(lambda a,b : a[b], 2, 1)
                #top_class_emb_array = map_to_train_emb_array(train_emb_array, top_N_class_label_indices)

                #map_to_train_emb_array = np.frompyfunc(lambda b : train_emb_array[b], 1, 1)

                #top_class_emb_array = map_to_train_emb_array(top_N_class_label_indices)
                top_class_emb_array =  np.zeros((top_N_class_label_indices.shape[0], top_N_class_label_indices.shape[1], 512))
                for i in range(top_N_class_label_indices.shape[0]):
                    for j in range(top_N_class_label_indices.shape[1]):
                        top_class_emb_array[i, j, :] = train_emb_array[top_N_class_label_indices[i][j]]
                
                #bp()
                #for row in train_emb_array:
                    #mapped_row = []
                
                # top_class_emb_array = train_emb_array[
                #     np.repeat(np.expand_dims(np.arange(train_emb_array.shape[0]), axis=1), N, axis=1),
                #     top_N_class_label_indices
                #]

                top_class_distances = np.expand_dims(emb_array, axis=1)
                top_class_distances = top_class_distances - top_class_emb_array
                top_class_distances = np.linalg.norm(top_class_distances, axis=2)

                top_1_class_distance_indices = np.argmin(top_class_distances, axis=1)
                top_1_class_label_indices = top_N_class_label_indices[
                    np.arange(top_N_class_label_indices.shape[0]),
                    top_1_class_distance_indices
                ]
                
                # best_class_indices = np.argmax(predictions, axis=1)
                best_class_indices = top_1_class_label_indices
                best_class_names = np.array([combined_class_names[idx] for idx in best_class_indices])
                
                #best_class_names = np.arang;  # gw: todo: use ix_ to make a ndarray with row being test_indices, and col (only 1) being the best class name
                # best_class_indices = np.argsort(predictions, axis=1)[-8:] #gw: todo, calcuate embedding distance again amont 8 rfc results
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                

                

                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, combined_class_names[best_class_indices[i]], best_class_probabilities[i]))

                # accuracy = np.mean(np.equal(best_class_indices, label_numbers))
                # accuracy = np.mean(np.equal(best_class_indices, label_names))
                bp()
                accuracy = np.mean(arr_eq(best_class_names, label_names))
                print('Accuracy: %.13f' % accuracy)


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

# python src/classifier_multi.py CLASSIFY --data_dir /home/gaopeng/workspace/ms1m-aligned-full/gw_celeb_3500_20_val/ --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..9}_rfc_classifier.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 --use_split_dataset
# python src/classifier_multi_2nd_cmpr_classify.py CLASSIFY --data_dir /home/gaopeng/workspace/ms1m-aligned-full/gw_celeb_3500_20_val/ --model /home/gaopeng/models/facenet/20180402-114759/20180402-114759.pb --classifier_filename ~/models/gw_subset_{1..8}_svc_classifier_with_emb.pkl --batch_size 1536 --min_nrof_images_per_class 20 --nrof_train_images_per_class 15 --use_split_dataset
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--classifier_filename',
                        nargs='*',
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
