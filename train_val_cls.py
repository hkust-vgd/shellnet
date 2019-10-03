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

LOG_FOUT = None
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_folder', '-s', default='log/cls/', help='Path to folder for saving check points and summary')
    parser.add_argument('--model', '-m', default='shellconv', help='Model to use')
    parser.add_argument('--setting', '-x', default='cls_modelnet40', help='Setting to use')
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
    batch_size_val = setting.batch_size_val
    sample_num = setting.sample_num

    # Prepare inputs
    print('{}-Preparing datasets...'.format(datetime.now()))
    [data_train, label_train] = provider.load_cls_files(setting.filelist)
    [data_val, label_val] = provider.load_cls_files(setting.filelist_val)
    data_train = data_train[:,0:sample_num,:]
    data_val = data_val[:,0:sample_num,:]
    label_train = np.squeeze(label_train)
    label_val = np.squeeze(label_val)
    num_train = label_train.shape[0]
    num_val = label_val.shape[0]
    num_batch_train = int((num_train+batch_size) / batch_size)
    num_batch_val = int(num_val / batch_size_val)

    num_extra = num_batch_train * batch_size - num_train
    choices = np.random.choice(num_train, num_extra, replace=False)
    data_extra = data_train[choices, :, :]
    label_extra = label_train[choices]
    data_train = np.concatenate((data_train, data_extra), 0)
    label_train = np.concatenate((label_train, label_extra), 0)

    ######################################################################
 
    global_step = tf.Variable(0, trainable=False, name='global_step')
    is_training_pl = tf.placeholder(tf.bool, name='is_training')

    data_pl = tf.placeholder(tf.float32, [None, sample_num, setting.data_dim], name='data_pl')
    label_pl = tf.placeholder(tf.int64, [None], name='label_pl')
    
    bn_decay_exp_op = tf.train.exponential_decay(0.5, global_step*batch_size, setting.decay_steps,
                                           0.5, staircase=True)
    bn_decay_op = tf.minimum(0.99, 1 - bn_decay_exp_op)
    logits_op = model.get_model(data_pl, is_training_pl, setting.sconv_params, setting.sdconv_params, setting.fc_params, 
                                weight_decay=setting.weight_decay, 
                                bn_decay = bn_decay_op, 
                                part_num=setting.num_class)

    # compute loss
    if setting.with_multi:
        labels_2d = tf.expand_dims(label_pl, axis=-1, name='labels_2d')
        label_pl = tf.tile(labels_2d, (1, logits_op.shape[1]), name='labels_tile')
        predictions_op = tf.argmax(tf.reduce_mean(logits_op, axis = -2), axis=-1, name='predictions')
    else:
        predictions_op = tf.argmax(logits_op, axis=-1, name='predictions')
    
    loss_op = tf.losses.sparse_softmax_cross_entropy(labels=label_pl, logits=logits_op)

    lr_exp_op = tf.train.exponential_decay(setting.learning_rate_base, global_step*batch_size, setting.decay_steps,
                                           setting.decay_rate, staircase=True)
    lr_clip_op = tf.maximum(lr_exp_op, setting.learning_rate_min)
    _ = tf.summary.scalar('learning_rate', tensor=lr_clip_op, collections=['train'])
    
    if setting.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_clip_op)
    elif setting.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=lr_clip_op, momentum=setting.momentum, use_nesterov=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss_op, global_step=global_step)

    saver = tf.train.Saver(max_to_keep=None)

    folder_ckpt = os.path.join(root_folder, 'ckpts')
    if not os.path.exists(folder_ckpt):
        os.makedirs(folder_ckpt)

    folder_summary = os.path.join(root_folder, 'summary')
    if not os.path.exists(folder_summary):
        os.makedirs(folder_summary)

    parameter_num = np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()])
    print('{}-Parameter number: {:d}.'.format(datetime.now(), parameter_num))

    shape_names = [line.rstrip() for line in \
        open(setting.shape_name_file)] 

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    # set config to save more gpu memory 
    with tf.Session(config=config) as sess:
        summaries_op = tf.summary.merge_all('train')
        summary_writer = tf.summary.FileWriter(folder_summary, sess.graph)

        sess.run(tf.global_variables_initializer())

        best_acc = 0
        best_epoch = 0
        start_time = time()
        for epoch in range(num_epochs):
            log_string('\n----------------------------- EPOCH %03d -----------------------------' % (epoch))
            
            # train
            total_correct = 0
            total_seen = 0    
            loss_avg = 0.
            data_train, label_train = provider.shuffle_data(data_train, label_train)
            for batch_idx in range(num_batch_train):
                start_time_sub = time()

                start_idx = batch_idx * batch_size
                end_idx = (batch_idx+1) * batch_size
                if setting.with_multi:
                    labels_2d = np.expand_dims(label_train[start_idx:end_idx], axis=-1)
                    labels_cur = np.tile(labels_2d, (1, setting.sconv_params[-1]['P']))
                else:
                    labels_cur = label_train[start_idx:end_idx]

                feed_dict = {data_pl: data_train[start_idx:end_idx, :, :],
                            label_pl: labels_cur,
                            is_training_pl: True,}
                
                _, loss, predictions, summaries, step = sess.run([train_op, loss_op, predictions_op, summaries_op, global_step], feed_dict=feed_dict)

                summary_writer.add_summary(summaries, step)

                correct = np.sum(predictions == label_train[start_idx:end_idx])
                total_correct += correct
                total_seen += batch_size

                loss_avg = loss_avg*batch_idx/(batch_idx+1) + loss/(batch_idx+1)
                print('[ep:%d/%d: %d/%d] train acc(oa): %f -- loss: %f -- time cost: %f' % \
                    (epoch, num_epochs, batch_idx, num_batch_train, total_correct/float(total_seen), loss_avg, (time() - start_time_sub)))
            
            log_string('training -- acc(oa): %f -- loss: %f ' % (total_correct / float(total_seen), loss_avg))
            log_string("Epoch %d, time cost: %f" % (epoch, time() - start_time))

            # Validation
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            total_seen_class = [0 for _ in range(setting.num_class)]
            total_correct_class = [0 for _ in range(setting.num_class)]
    
            eval_start_time = time()  # eval start time
            for batch_idx in range(num_batch_val):
                start_idx = batch_idx * batch_size_val
                end_idx = (batch_idx+1) * batch_size_val
                if setting.with_multi:
                    labels_2d = np.expand_dims(label_val[start_idx:end_idx], axis=-1)
                    labels_cur = np.tile(labels_2d, (1, setting.sconv_params[-1]['P']))
                else:
                    labels_cur = label_val[start_idx:end_idx]
         
                feed_dict = {data_pl: data_val[start_idx:end_idx, :, :],
                            label_pl: labels_cur,
                            is_training_pl: False,}

                loss, predictions_val = sess.run([loss_op, predictions_op], feed_dict=feed_dict)

                correct = np.sum(predictions_val == label_val[start_idx:end_idx])
                total_correct += correct
                total_seen += batch_size_val
                loss_sum += (loss*batch_size_val)
                for i in range(start_idx, end_idx):
                    l = label_val[i]
                    total_seen_class[l] += 1
                    total_correct_class[l] += (predictions_val[i-start_idx] == l)

            class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
            acc_mean_cls = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
            acc_oa = total_correct / float(total_seen)
            loss_val = loss_sum / float(total_seen)

            log_string('eval acc (oa): %f ---- eval acc (mean class): %f ---- loss: %f' % \
                (acc_oa, acc_mean_cls, loss_val))

            print('eval time: %f' % (time() - eval_start_time))
            if acc_oa > best_acc:
                save_path = saver.save(sess, os.path.join(folder_ckpt, "best_model_epoch.ckpt"))
                best_acc = acc_oa
                best_epoch = epoch

                with open(os.path.join(root_folder,"class_accuracies.txt"), 'w') as the_file:
                    the_file.write('best epoch: %f \n' % (best_epoch))
                    for i, name in enumerate(shape_names):
                        print('%10s:\t%0.3f' % (name, class_accuracies[i]))
                        the_file.write('%10s:\t%0.3f\n' % (name, class_accuracies[i]))
                    the_file.write('%10s:\t%0.3f\n' % ('mean class acc: ', acc_mean_cls))
            
if __name__ == '__main__':
    main()
    if LOG_FOUT is not None:
        LOG_FOUT.close()