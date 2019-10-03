#!/usr/bin/python3
"""Training and Validation On Segmentation Task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import random
import shutil
import argparse
import importlib
import numpy as np
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'setting'))
import data_utils
import pointfly as pf
import provider

LOG_FOUT = None
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', '-l', help='Path to a check point file for load')
    parser.add_argument('--save_folder', '-s', default='log/seg', help='Path to folder for saving check points and summary')
    parser.add_argument('--model', '-m', default='shellconv', help='Model to use')
    parser.add_argument('--setting', '-x', default='seg_s3dis', help='Setting to use')
    parser.add_argument('--log', help='Log to FILE in save folder; use - for stdout (default is log.txt)', metavar='FILE', default='log.txt')
    args = parser.parse_args()

    time_string = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    root_folder = os.path.join(args.save_folder, '%s_%s_%s' % (args.model, args.setting, time_string))
    
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    global LOG_FOUT
    if args.log != '-':
        LOG_FOUT = open(os.path.join(root_folder, args.log), 'w')

    model = importlib.import_module(args.model)
    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    num_epochs = setting.num_epochs
    batch_size = setting.batch_size
    sample_num = setting.sample_num
    step_val = setting.step_val
    label_weights_list = setting.label_weights
    rotation_range = setting.rotation_range
    rotation_range_val = setting.rotation_range_val
    scaling_range = setting.scaling_range
    scaling_range_val = setting.scaling_range_val
    jitter = setting.jitter
    jitter_val = setting.jitter_val    

    is_list_of_h5_list = data_utils.is_h5_list(setting.filelist)
    if is_list_of_h5_list:
        seg_list = [setting.filelist] # for train
    else:
        seg_list = data_utils.load_seg_list(setting.filelist)  # for train
    data_val, _, data_num_val, label_val, _ = data_utils.load_seg(setting.filelist_val)
    if data_val.shape[-1] > 3:    
        data_val = data_val[:,:,:3]  # only use the xyz coordinates
    point_num = data_val.shape[1]
    num_val = data_val.shape[0]
    batch_num_val = num_val // batch_size

    ######################################################################
    # Placeholders
    indices = tf.placeholder(tf.int32, shape=(None, sample_num, 2), name="indices")
    xforms = tf.placeholder(tf.float32, shape=(None, 3, 3), name="xforms")
    rotations = tf.placeholder(tf.float32, shape=(None, 3, 3), name="rotations")
    jitter_range = tf.placeholder(tf.float32, shape=(1), name="jitter_range")
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training = tf.placeholder(tf.bool, name='is_training')

    pts_fts = tf.placeholder(tf.float32, shape=(None, point_num, setting.data_dim), name='pts_fts')
    labels_seg = tf.placeholder(tf.int64, shape=(None, point_num), name='labels_seg')
    labels_weights = tf.placeholder(tf.float32, shape=(None, point_num), name='labels_weights')

    ######################################################################
    points_sampled = tf.gather_nd(pts_fts, indices=indices, name='pts_fts_sampled')
    points_augmented = pf.augment(points_sampled, xforms, jitter_range)

    labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
    labels_weights_sampled = tf.gather_nd(labels_weights, indices=indices, name='labels_weight_sampled')

    bn_decay_exp_op = tf.train.exponential_decay(0.5, global_step, setting.decay_steps,
                                           0.5, staircase=True)
    bn_decay_op = tf.minimum(0.99, 1 - bn_decay_exp_op)

    logits_op = model.get_model(points_augmented, is_training, setting.sconv_params, setting.sdconv_params, setting.fc_params, 
                            sampling=setting.sampling,
                            weight_decay=setting.weight_decay, 
                            bn_decay = bn_decay_op, 
                            part_num=setting.num_class)

    predictions = tf.argmax(logits_op, axis=-1, name='predictions')

    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels_sampled, logits=logits_op,
                                                     weights=labels_weights_sampled)

    with tf.name_scope('metrics'):
        loss_mean_op, loss_mean_update_op = tf.metrics.mean(loss_op)
        t_1_acc_op, t_1_acc_update_op = tf.metrics.accuracy(labels_sampled, predictions, weights=labels_weights_sampled)
        t_1_per_class_acc_op, t_1_per_class_acc_update_op = \
            tf.metrics.mean_per_class_accuracy(labels_sampled, predictions, setting.num_class,
                                               weights=labels_weights_sampled)
    reset_metrics_op = tf.variables_initializer([var for var in tf.local_variables()
                                                 if var.name.split('/')[0] == 'metrics'])


    _ = tf.summary.scalar('loss/train', tensor=loss_mean_op, collections=['train'])
    _ = tf.summary.scalar('t_1_acc/train', tensor=t_1_acc_op, collections=['train'])
    _ = tf.summary.scalar('t_1_per_class_acc/train', tensor=t_1_per_class_acc_op, collections=['train'])

    _ = tf.summary.scalar('loss/val', tensor=loss_mean_op, collections=['val'])
    _ = tf.summary.scalar('t_1_acc/val', tensor=t_1_acc_op, collections=['val'])
    _ = tf.summary.scalar('t_1_per_class_acc/val', tensor=t_1_per_class_acc_op, collections=['val'])

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
                                           
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])

    reg_loss = setting.weight_decay * tf.losses.get_regularization_loss()
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op + reg_loss, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=None)

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.Session(config=config) as sess:
        summaries_op = tf.summary.merge_all('train')
        summaries_val_op = tf.summary.merge_all('val')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(tf.global_variables_initializer())

        # Load the model
        if args.load_ckpt is not None:
            saver.restore(sess, args.load_ckpt)
            print('{}-Checkpoint loaded from {}!'.format(datetime.now(), args.load_ckpt))
        else:
            latest_ckpt = tf.train.latest_checkpoint(folder_ckpt)
            if latest_ckpt:
                print('{}-Found checkpoint {}'.format(datetime.now(), latest_ckpt))
                saver.restore(sess, latest_ckpt)
                print('{}-Checkpoint loaded from {} (Iter {})'.format(
                    datetime.now(), latest_ckpt, sess.run(global_step)))

        best_acc = 0
        best_epoch = 0
        for epoch in range(num_epochs):
            ############################### train #######################################
            # Shuffle train files
            np.random.shuffle(seg_list)
            for file_idx_train in range(len(seg_list)):
                print('----epoch:'+str(epoch) + '--train file:' + str(file_idx_train) + '-----')
                filelist_train = seg_list[file_idx_train]
                data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
                num_train = data_train.shape[0]
                if data_train.shape[-1] > 3:    
                    data_train = data_train[:,:,:3]
                data_train, data_num_train, label_train = \
                    data_utils.grouped_shuffle([data_train, data_num_train, label_train])
                # data_train, label_train, _ = provider.shuffle_data_seg(data_train, label_train) 

                batch_num = (num_train + batch_size - 1) // batch_size

                for batch_idx_train in range(batch_num):
                    # Training
                    start_idx = (batch_size * batch_idx_train) % num_train
                    end_idx = min(start_idx + batch_size, num_train)
                    batch_size_train = end_idx - start_idx
                    points_batch = data_train[start_idx:end_idx, ...]
                    points_num_batch = data_num_train[start_idx:end_idx, ...]
                    labels_batch = label_train[start_idx:end_idx, ...]
                    weights_batch = np.array(label_weights_list)[labels_batch]

                    offset = int(random.gauss(0, sample_num * setting.sample_num_variance))
                    offset = max(offset, -sample_num * setting.sample_num_clip)
                    offset = min(offset, sample_num * setting.sample_num_clip)
                    sample_num_train = sample_num + offset
                    xforms_np, rotations_np = pf.get_xforms(batch_size_train,
                                                            rotation_range=rotation_range,
                                                            scaling_range=scaling_range,
                                                            order=setting.rotation_order)
                    sess.run(reset_metrics_op)
                    sess.run([train_op, loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op],
                            feed_dict={
                                pts_fts: points_batch,
                                indices: pf.get_indices(batch_size_train, sample_num_train, points_num_batch),
                                xforms: xforms_np,
                                rotations: rotations_np,
                                jitter_range: np.array([jitter]),
                                labels_seg: labels_batch,
                                labels_weights: weights_batch,
                                is_training: True,
                            })
                
                loss, t_1_acc, t_1_per_class_acc, summaries, step = sess.run([loss_mean_op,
                                                                        t_1_acc_op,
                                                                        t_1_per_class_acc_op,
                                                                        summaries_op,
                                                                        global_step])
                summary_writer.add_summary(summaries, step)
                log_string('{}-[Train]-Iter: {:06d}  Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f}'
                    .format(datetime.now(), step, loss, t_1_acc, t_1_per_class_acc))
                sys.stdout.flush()
                ######################################################################
        
            filename_ckpt = os.path.join(folder_ckpt, 'epoch')
            saver.save(sess, filename_ckpt, global_step=epoch)
            print('{}-Checkpoint saved to {}!'.format(datetime.now(), filename_ckpt))

            sess.run(reset_metrics_op)
            for batch_val_idx in range(batch_num_val):
                start_idx = batch_size * batch_val_idx
                end_idx = min(start_idx + batch_size, num_val)
                batch_size_val = end_idx - start_idx
                points_batch = data_val[start_idx:end_idx, ...]
                points_num_batch = data_num_val[start_idx:end_idx, ...]
                labels_batch = label_val[start_idx:end_idx, ...]
                weights_batch = np.array(label_weights_list)[labels_batch]

                xforms_np, rotations_np = pf.get_xforms(batch_size_val,
                                                            rotation_range=rotation_range_val,
                                                            scaling_range=scaling_range_val,
                                                            order=setting.rotation_order)
                sess.run([loss_mean_update_op, t_1_acc_update_op, t_1_per_class_acc_update_op],
                        feed_dict={
                            pts_fts: points_batch,
                            indices: pf.get_indices(batch_size_val, sample_num, points_num_batch),
                            xforms: xforms_np,
                            rotations: rotations_np,
                            jitter_range: np.array([jitter_val]),
                            labels_seg: labels_batch,
                            labels_weights: weights_batch,
                            is_training: False,
                        })
            loss_val, t_1_acc_val, t_1_per_class_acc_val, summaries_val, step = sess.run(
                [loss_mean_op, t_1_acc_op, t_1_per_class_acc_op, summaries_val_op, global_step])
            summary_writer.add_summary(summaries_val, step)

            if t_1_per_class_acc_val > best_acc:
                best_acc = t_1_per_class_acc_val
                best_epoch = epoch

            log_string('{}-[Val  ]-Average:      Loss: {:.4f}  T-1 Acc: {:.4f}  T-1 mAcc: {:.4f} best epoch: {} Current epoch: {}'
                  .format(datetime.now(), loss_val, t_1_acc_val, t_1_per_class_acc_val, best_epoch, epoch))
            sys.stdout.flush()
            ######################################################################
            
        print('{}-Done!'.format(datetime.now()))

if __name__ == '__main__':
    main()
    if LOG_FOUT is not None:
        LOG_FOUT.close()