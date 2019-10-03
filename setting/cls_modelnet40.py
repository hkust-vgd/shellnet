filelist = '../data/modelnet/train_files.txt'
filelist_val = '../data/modelnet/test_files.txt'
shape_name_file = '../data/modelnet/shape_names.txt'

num_class = 40
sample_num = 1024
batch_size = 32
batch_size_val = 4
num_epochs = 250
step_val = 1000
momentum = 0.9

learning_rate_base = 0.001
decay_steps = 200000
decay_rate = 0.7
learning_rate_min = 1e-5

weight_decay = 1e-8 # 'wd':0.0001  # weight_decay = 1e-6  to avoid overfitting

ss = 16 # shell size (number of points contained in each shell)
sconv_param_name = ('K', 'D', 'P', 'C')
sconv_params = [dict(zip(sconv_param_name, sconv_param)) for sconv_param in
                [
                 (ss*4, 4, 512, 128),
                 (ss*2, 2, 128, 256),
                 (ss*1, 1, 32, 512)]]

sdconv_params = None

x=2

fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
            [(128 * x, 0.2),
            (64 * x, 0.5)]]
sampling = 'random' # 'fps' or 'random'
optimizer = 'adam'

data_dim = 3
with_multi=True