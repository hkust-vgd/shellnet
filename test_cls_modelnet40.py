import os
import sys
import argparse
import importlib
import numpy as np
from time import time
import tensorflow as tf
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'setting'))
from utils import provider

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_ckpt', '-l', default='log/cls/pretrained/ckpts/', help='Path to a check point file for load')
    parser.add_argument('--model', '-m', default='shellconv', help='Model to use')
    parser.add_argument('--setting', '-x', default='cls_modelnet40', help='Setting to use')
    args = parser.parse_args()

    setting_path = os.path.join(os.path.dirname(__file__), args.model)
    sys.path.append(setting_path)
    setting = importlib.import_module(args.setting)

    batch_size_val = setting.batch_size_val
    sample_num = setting.sample_num

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    [data_val, label_val] = provider.load_cls_files(setting.filelist_val)
    data_val = data_val[:,0:sample_num,:]
    label_val = np.squeeze(label_val)
    num_val = label_val.shape[0]
    num_batch_val = int(num_val / batch_size_val)

    # load shape names
    shape_names = [line.rstrip() for line in \
        open(setting.shape_name_file)] 
    
    ckpt = tf.train.get_checkpoint_state(args.load_ckpt)
    saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
    graph = tf.get_default_graph()   
    ops = {'data_pl': graph.get_tensor_by_name("data_pl:0"),
           'is_training_pl': graph.get_tensor_by_name("is_training:0"),
           'predictions_op': graph.get_tensor_by_name("predictions:0")}

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False        
    sess = tf.Session(config=config)  

    saver.restore(sess, ckpt.model_checkpoint_path)

    # testing
    total_correct = 0
    total_seen = 0
    total_seen_class = [0 for _ in range(setting.num_class)]
    total_correct_class = [0 for _ in range(setting.num_class)]
    
    eval_start_time = time()  # eval start time
    for batch_idx in range(num_batch_val):
        start_idx = batch_idx * batch_size_val
        end_idx = (batch_idx+1) * batch_size_val

        feed_dict = {ops['data_pl']: data_val[start_idx:end_idx, :, :],
                    ops['is_training_pl']: False,}
        
        # infer_start_time = time()
        predictions_val = sess.run(ops['predictions_op'], feed_dict=feed_dict)
        # print('infer time : %f' % (time() - infer_start_time))

        correct = np.sum(predictions_val == label_val[start_idx:end_idx])
        total_correct += correct
        total_seen += batch_size_val
        for i in range(start_idx, end_idx):
            l = label_val[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (predictions_val[i-start_idx] == l)

    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    acc_mean_cls = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    acc_oa = total_correct / float(total_seen)

    
    print('eval acc (oa): %f ---- eval acc (mean class): %f ---- time cost: %f' % \
        (acc_oa, acc_mean_cls, time() - eval_start_time))

    print('per-class accuracies:')
    for i, name in enumerate(shape_names):
        print('%10s:\t%0.3f' % (name, class_accuracies[i]))
            
if __name__ == '__main__':
    main()