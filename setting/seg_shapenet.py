#!/usr/bin/python3
import math

filelist = '../data/shapenet_partseg/train_val_files.txt'
filelist_val = '../data/shapenet_partseg/test_files.txt'

# 'Path to category list file (.txt)'
category = '../data/shapenet_partseg/categories.txt' 

# 'Path to *.pts directory'
data_folder = '../data/shapenet_partseg/test_data'  

num_class = 50

sample_num = 2048

batch_size = 16

num_epochs = 500

label_weights = [1.0] * num_class

learning_rate_base = 0.001
decay_steps = 20000
step_val = 500
decay_rate = 0.7
learning_rate_min = 0.000001

weight_decay = 0.0

jitter = 0.001
jitter_val = 0.0

rotation_range = [0, math.pi/32., 0, 'u']
rotation_range_val = [0, 0, 0, 'u']
rotation_order = 'rxyz'

scaling_range = [0.0, 0.0, 0.0, 'g']
scaling_range_val = [0, 0, 0, 'u']

sample_num_variance = 1 // 8
sample_num_clip = 1 // 4

ss = 8 # shell size (number of points contained in each shell)
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
            [(128 * x, 0.2),
            (64 * x, 0.2)]]

sampling = 'fps'  # 'fps' or 'random'

optimizer = 'adam'
epsilon = 1e-3

data_dim = 3

keep_remainder = True

sorting_method = None
with_global = True
with_local = True
