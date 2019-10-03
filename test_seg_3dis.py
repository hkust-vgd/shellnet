#!/usr/bin/python3
"""Testing On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'setting'))
from utils import provider
from utils import data_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', '-l', default='log/seg//shellconv_seg_s3dis_area_2/ckpts/epoch-19', help='Path to a check point file for load')
    parser.add_argument('--model', '-m', default='shellconv', help='Model to use')
    parser.add_argument('--setting', '-x', default='seg_s3dis', help='Setting to use')
    parser.add_argument('--repeat_num', '-r', help='Repeat number', type=int, default=1)
    parser.add_argument('--save_ply', '-s', help='Save results as ply', default=False)
    args = parser.parse_args(
    print(args)

    
    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    sample_num = setting.sample_num
    max_point_num = setting.max_point_num
    batch_size = args.repeat_num * math.ceil(max_point_num / sample_num)

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(batch_size, None, 2), name="indices")
    is_training = tf.placeholder(tf.bool, name='is_training')
    pointclouds = tf.placeholder(tf.float32, shape=(batch_size, max_point_num, setting.data_dim), name='points')
    ######################################################################

    ######################################################################
    points_sampled = tf.gather_nd(pointclouds, indices=indices, name='pts_sampled')
    

    logits_op = model.get_model(points_sampled, is_training, setting.sconv_params, setting.sdconv_params, setting.fc_params, 
                            sampling=setting.sampling,
                            weight_decay=setting.weight_decay, 
                            bn_decay = None, 
                            part_num=setting.num_class)
    probs_op = tf.nn.softmax(logits_op, name='probs_op')

    # for restore model
    saver = tf.train.Saver()

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    with tf.Session() as sess:
        # Load the model
        saver.restore(sess, args.load_ckpt)
        print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))

        indices_batch_indices = np.tile(np.reshape(np.arange(batch_size), (batch_size, 1, 1)), (1, sample_num, 1))

        folder = os.path.dirname(setting.filelist_val)
        filenames = [os.path.join(folder, line.strip()) for line in open(setting.filelist_val)]
        for filename in filenames:
            print('{}-Reading {}...'.format(datetime.now(), filename))
            data_h5 = h5py.File(filename)
            data = data_h5['data'][...].astype(np.float32)
            data_num = data_h5['data_num'][...].astype(np.int32)
            batch_num = data.shape[0]
            if data.shape[-1] > 3:
                data = data[:,:,:3]

            labels_pred = np.full((batch_num, max_point_num), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)

            print('{}-{:d} testing batches.'.format(datetime.now(), batch_num))
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    print('{}-Processing {} of {} batches.'.format(datetime.now(), batch_idx, batch_num))
                points_batch = data[[batch_idx] * batch_size, ...]
                point_num = data_num[batch_idx]

                tile_num = math.ceil((sample_num * batch_size) / point_num)
                indices_shuffle = np.tile(np.arange(point_num), tile_num)[0:sample_num * batch_size]
                np.random.shuffle(indices_shuffle)
                indices_batch_shuffle = np.reshape(indices_shuffle, (batch_size, sample_num, 1))
                indices_batch = np.concatenate((indices_batch_indices, indices_batch_shuffle), axis=2)

                seg_probs = sess.run([probs_op],
                                        feed_dict={
                                            pointclouds: points_batch,
                                            indices: indices_batch,
                                            is_training: False,
                                        })
                probs_2d = np.reshape(seg_probs, (sample_num * batch_size, -1))

                predictions = [(-1, 0.0)] * point_num
                for idx in range(sample_num * batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, :]
                    confidence = np.amax(probs)
                    label = np.argmax(probs)
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                labels_pred[batch_idx, 0:point_num] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num] = np.array([confidence for _, confidence in predictions])

            filename_pred = filename[:-3] + '_pred.h5'
            print('{}-Saving {}...'.format(datetime.now(), filename_pred))
            file = h5py.File(filename_pred, 'w')
            file.create_dataset('data_num', data=data_num)
            file.create_dataset('label_seg', data=labels_pred)
            file.create_dataset('confidence', data=confidences_pred)
            has_indices = 'indices_split_to_full' in data_h5
            if has_indices:
                file.create_dataset('indices_split_to_full', data=data_h5['indices_split_to_full'][...])
            file.close()

            if args.save_ply:
                print('{}-Saving ply of {}...'.format(datetime.now(), filename_pred))
                filepath_label_ply = os.path.join(filename_pred[:-3] + 'ply_label')
                data_utils.save_ply_property_batch(data[:, :, 0:3], labels_pred[...],
                                                   filepath_label_ply, data_num[...], setting.num_class)
            ######################################################################
        print('{}-Done!'.format(datetime.now()))


if __name__ == '__main__':
    main()