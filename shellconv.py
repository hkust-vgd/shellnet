import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'setting'))
import pointfly as pf

def dense(input, output, is_training, name, bn_decay=None, with_bn=True, activation=tf.nn.relu):
    if with_bn:
        input = tf.layers.batch_normalization(input, momentum=0.98, training=is_training, name=name+'bn')
    
    dense = tf.layers.dense(input, output, activation=activation, name=name)
    
    return dense
    
def conv2d(input, output, name, is_training, kernel_size, bn_decay=None, 
            reuse=None, with_bn=True, activation=tf.nn.relu):
    if with_bn:
        input = tf.layers.batch_normalization(input, momentum=0.98, training=is_training, name=name+'bn')
    
    conv2d = tf.layers.conv2d(input, output, kernel_size=kernel_size, strides=(1, 1), padding='VALID',
                              activation=activation, reuse=reuse, name=name, use_bias=not with_bn)
    return conv2d

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def shellconv(pts, fts_prev, qrs, is_training, tag, K, D, P, C, with_local, bn_decay=None):
    indices = pf.knn_indices_general(qrs, pts, K, True)

    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    [N,P,K,dim] = nn_pts_local.shape # (N, P, K, 3)
    nn_fts_local = None
    C_pts_fts = 64
    if with_local:
        nn_fts_local = dense(nn_pts_local, C_pts_fts // 2, is_training, tag + 'nn_fts_from_pts_0',bn_decay=bn_decay)
        nn_fts_local = dense(nn_fts_local, C_pts_fts, is_training, tag + 'nn_fts_from_pts',bn_decay=bn_decay)
    else:
        nn_fts_local = nn_pts_local

    if fts_prev is not None:
        fts_prev = tf.gather_nd(fts_prev, indices, name=tag + 'fts_prev')  # (N, P, K, 3)
        pts_X_0 = tf.concat([nn_fts_local,fts_prev], axis=-1)
    else:
        pts_X_0 = nn_fts_local

    s = int(K.value/D)  # no. of divisions
    feat_max = tf.layers.max_pooling2d(pts_X_0, [1,s], strides=[1,s], padding='valid', name=tag+'maxpool_0')

    fts_X = conv2d(feat_max, C, name=tag+'conv', is_training=is_training, kernel_size=[1,feat_max.shape[-2].value])

    fts_X = tf.squeeze(fts_X, axis=-2)
    return fts_X
    
def get_model(layer_pts, is_training, sconv_params, sdconv_params, fc_params, sampling='random', weight_decay=0.0, bn_decay=None, part_num=8):
    if sampling == 'fps':
        sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
        from tf_sampling import farthest_point_sample, gather_point

    layer_fts_list = [None]
    layer_pts_list = [layer_pts]
    for layer_idx, layer_param in enumerate(sconv_params):
        tag = 'sconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        if P == -1:
            qrs = layer_pts
        else:
            if sampling == 'fps':
                qrs = gather_point(layer_pts, farthest_point_sample(P, layer_pts))
            elif sampling == 'random':
                qrs = tf.slice(layer_pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            else:
                print('Unknown sampling method!')
                exit()

        layer_fts= shellconv(layer_pts_list[-1], layer_fts_list[-1], qrs, is_training, tag, K, D, P, C, True, bn_decay)

        layer_pts = qrs
        layer_pts_list.append(qrs)
        layer_fts_list.append(layer_fts)
  
    if sdconv_params is not None:
        fts = layer_fts_list[-1]
        for layer_idx, layer_param in enumerate(sdconv_params):
            tag = 'sdconv_' + str(layer_idx + 1) + '_'
            K = layer_param['K'] 
            D = layer_param['D'] 
            pts_layer_idx = layer_param['pts_layer_idx']  # 2 1 0 
            qrs_layer_idx = layer_param['qrs_layer_idx']  # 1 0 -1

            pts = layer_pts_list[pts_layer_idx + 1]
            qrs = layer_pts_list[qrs_layer_idx + 1]
            fts_qrs = layer_fts_list[qrs_layer_idx + 1]

            C = fts_qrs.get_shape()[-1].value if fts_qrs is not None else C//2
            P = qrs.get_shape()[1].value
            
            layer_fts= shellconv(pts, fts, qrs, is_training, tag, K, D, P, C, True, bn_decay)
            if fts_qrs is not None: # this is for last layer
                fts_concat = tf.concat([layer_fts, fts_qrs], axis=-1, name=tag + 'fts_concat')
                fts = dense(fts_concat, C, is_training, tag + 'mlp', bn_decay=bn_decay)           
            
    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        layer_fts = dense(layer_fts, C, is_training, name='fc{:d}'.format(layer_idx), bn_decay=bn_decay)
        layer_fts = tf.layers.dropout(layer_fts, rate=dropout_rate, name='fc{:d}_dropout'.format(layer_idx))
    
    logits = dense(layer_fts, part_num, is_training, name='logits', activation=None)
    return logits

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.device('/gpu:0'):
            sconv_param_name = ('K', 'D', 'P', 'C', 'links')
            sconv_params = [dict(zip(sconv_param_name, sconv_param)) for sconv_param in
                            [(32, 4, 512, 128, []),
                             (16, 2, 256, 256, []),
                             (8, 1, 128, 512, [])]]

            sdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
            sdconv_params = [dict(zip(sdconv_param_name, sdconv_param)) for sdconv_param in
                            [(8, 1, 2, 1),
                             (16, 2, 1, 0),
                             (32, 4, 0, -1)]]

            x = 2
            fc_param_name = ('C', 'dropout_rate')
            fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                        [(128 * x, 0),
                        (64 * x, 0.5)]]
            inputs = tf.random_uniform([32, 1024, 3], minval=0, maxval=10,dtype=tf.float32)

            with_local = True
            
            outputs = get_model(inputs, tf.constant(True), sconv_params, sdconv_params, fc_params)
            print(outputs)
