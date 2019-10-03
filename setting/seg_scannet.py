#!/usr/bin/python3
import math

filelist = '../data/scannet/seg/train_files.txt'
filelist_val = '../data/scannet/seg/test_files.txt'

num_class = 21

sample_num = 2048

max_point_num = 8192

batch_size = 16

num_epochs = 200

label_weights = [0.0] * 1 + [1.0] * (num_class - 1)

learning_rate_base = 0.005
decay_steps = 5000
step_val = 500
decay_rate = 0.8 
learning_rate_min = 1e-6

weight_decay = 1e-8

jitter = 0.001
jitter_val = 0.0

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.001, 0.001, 0.001, 'g']
scaling_range_val = [0, 0, 0, 'u']

ss = 8  # shell size (number of points contained in each shell)
sconv_param_name = ('K', 'D', 'P', 'C')
sconv_params = [dict(zip(sconv_param_name, sconv_param)) for sconv_param in
                [
                 (ss*4, 4, 512, 128),
                 (ss*2, 2, 128, 256),
                 (ss*1, 1, 32, 512)]]

sdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
sdconv_params = [dict(zip(sdconv_param_name, sdconv_param)) for sdconv_param in
                [
                (ss*1,  1, 2, 1),
                (ss*2,  2, 1, 0),
                (ss*4,  4, 0, -1)]]

x=1
fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
            [(128 * x, 0.0),
            (64 * x, 0.2)]]

sampling = 'fps' # 'fps' or 'random'

optimizer = 'adam'

data_dim = 3