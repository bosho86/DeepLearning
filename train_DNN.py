import datetime
import os
import csv
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from numpy import fft
import pywt
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from config import train_config

from model import DNNModel

def load_data(config):
    print('Loading data from {} ...'.format(config['data_dir']))
    data = np.load('../data/data_train.npy')
    data.astype(np.float)
    labels = np.load('../data/labels_train.npy')
    labels.astype(np.float)
    # selecting the samples from data_train to be used for cross-validation
    num_cv_samples = int(config['cross_validation_samples']*data.shape[1])
    cv_samples = random.sample(range(data.shape[1]), num_cv_samples)
    data_cv = data[:,cv_samples,:]
    labels_cv = labels[:, cv_samples,:]
    data_train = np.delete(data, cv_samples, axis=1)
    labels_train = np.delete(labels, cv_samples, axis=1)

    data_test = np.load('../data/data_test.npy')
    data_test.astype(np.float)
    labels_test = np.load('../data/labels_test.npy')
    labels_test.astype(np.float)
    return [data_train, data_cv, data_test, labels_train, labels_cv, labels_test]

def data_init(config):
    print('Loading data from {} ...'.format(config['data_dir']))
    # read labels which are stored "one label on each line" format
    labels_csv_file = os.path.join(config['data_dir'], 'train_labels_3.csv')
    labels_file = open(labels_csv_file, 'r')
    labels_class = labels_file.readlines()
    labels_class = np.array(labels_class, dtype=int)
    labels_class = ((labels_class>0)*1)
    print(labels_class)
    labels_file.close()
    # convert labels to target class probabilities (i.e. one-hot encoding)
    labels = np.zeros((len(labels_class), 2))
    labels[list(range(len(labels_class))),labels_class] = 1
    # read training data in npy file format
    data_file = os.path.join(config['data_dir'], 'train_data_3.npy')
    data = np.load(data_file)
    data = data.astype(np.float)
    
    labels_delete = np.array(np.flatnonzero((labels_class>2)*1))
    labels = np.delete(labels,labels_delete,axis=0)
    data = np.delete(data,labels_delete,axis=0)
    

    
    # setting batch size to 1
    data  = np.expand_dims(data, axis=0)
    labels  = np.expand_dims(labels, axis=0)
    
    # selecting the samples from data_train to be used for cross-validation
    num_test_samples = int(0.16*data.shape[1])
    test_samples = random.sample(range(data.shape[1]), num_test_samples)
    data_test = data[:,test_samples,:]
    labels_test = labels[:, test_samples,:]
    np.save('../data/data_test.npy',data_test[:,:,:])
    np.save('../data/labels_test.npy',labels_test[:,:,:])
    data = np.delete(data, test_samples, axis=1)
    labels = np.delete(labels, test_samples, axis=1)
    np.save('../data/data_train.npy',data[:,:,:])
    np.save('../data/labels_train.npy',labels[:,:,:])
    num_cv_samples = int(config['cross_validation_samples']*data.shape[1])
    cv_samples = random.sample(range(data.shape[1]), num_cv_samples)
    data_cv = data[:,cv_samples,:]
    labels_cv = labels[:, cv_samples,:]
    data_train = np.delete(data, cv_samples, axis=1)
    labels_train = np.delete(labels, cv_samples, axis=1)
    return [data_train, data_cv, data_test, labels_train, labels_cv, labels_test]

def preprocess_data(data):
    # normalizing the data
    data = data[:,:,1024:5120]
    dataMean =data.mean(axis=-1)
    dataStd = np.std(data, axis=-1)
    data = (data - np.tile(np.expand_dims(dataMean, axis=-1),(1,1,data.shape[-1])))/np.tile(np.expand_dims(dataStd,axis=-1), (1,1,data.shape[-1]))
    # wavelet transform and back-transform to smoothen the signal
    w = pywt.Wavelet('db6')
    coeffs = pywt.wavedec(data, w, axis=-1)
    data = pywt.waverec(coeffs[:-2]+[None]*2, w, axis=-1)
    #data =  np.diff(data, n=5)
    print('starting FFT')
    fftX = fft.rfft(data, n=None, axis=2)
    print('FFT finished')
    X = np.abs(fftX)
    smoothenX = X
    window = 5
    for i in range(window,X.shape[2]-window):
        smoothenX[:, :, i] = np.sum(X[:, :, i-window:i+window], axis=2)/2/window
    print('Smoothening finished')
    smoothenX = smoothenX[:, :, ::4]
    return smoothenX

def get_model_and_placeholders(config):
    # create placeholders that we need to feed the required data into the model
    # None means that the dimension is variable, which we want for the batch size and the sequence length
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    
    input_pl = tf.placeholder(tf.float32, shape=[None, input_dim], name='input_pl')
    target_pl = tf.placeholder(tf.float32, shape=[None, output_dim], name='target_pl')

    placeholders = {'input_pl': input_pl,
                    'target_pl': target_pl}

    rnn_model_class = DNNModel
    return rnn_model_class, placeholders


def main(config):
    # create unique output directory for this model
    timestamp = str(int(time.time()))
    config['name'] = 'ECG_class_DNN_nodiff_wt-2_window5'
    config['model_dir'] = os.path.abspath(os.path.join(config['output_dir'], config['name']))
    try:
        os.makedirs(config['model_dir'])
    except Exception:
        print('Model dir exists already.')
    print('Writing checkpoints into {}'.format(config['model_dir']))

    # load the data, this requires that the *.npz files you downloaded from Kaggle be named `train.npz` and `valid.npz`
    [data_train, data_cv, data_test, labels_train, labels_cv, labels_test] = load_data(config)
    
    print(data_train.shape)
    print(labels_train.shape)
    print(data_test.shape)
     
    # preprocessing the data   
    data_train = preprocess_data(data_train)
    data_cv = preprocess_data(data_cv)
    data_test = preprocess_data(data_test)

    config['input_dim'] = data_train.shape[-1]
    config['output_dim'] = labels_train.shape[-1]
    # get input placeholders and get the model that we want to train
    rnn_model_class, placeholders = get_model_and_placeholders(config)

    # Create a variable that stores how many training iterations we performed.
    # This is useful for saving/storing the network
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # create a training graph, this is the graph we will use to optimize the parameters
    with tf.name_scope('training'):
        rnn_model = rnn_model_class(config, placeholders, mode='training')
        rnn_model_valid = rnn_model_class(config, placeholders, mode='validation')
        rnn_model_test = rnn_model_class(config, placeholders, mode='inference')
        rnn_model.build_graph()
        rnn_model_valid.build_graph()
        rnn_model_test.build_graph()
        print('created RNN model with {} parameters'.format(rnn_model.n_parameters))
        lr = config['learning_rate']
        # configure learning rate
        params = tf.trainable_variables()
        train_op = tf.train.AdamOptimizer(
                    learning_rate=lr,
                ).minimize(
                    loss=rnn_model.loss,
                    var_list=params,
                    name='adam',
                )



    # Create summary ops for monitoring the training
    # Each summary op annotates a node in the computational graph and collects data data from it
    tf.summary.scalar('learning_rate', lr, collections=['training_summaries'])

    # Merge summaries used during training and reported after every step
    summaries_training = tf.summary.merge(tf.get_collection('training_summaries'))

    # create summary ops for monitoring the validation
    # caveat: we want to store the performance on the entire validation set, not just one validation batch
    # Tensorflow does not directly support this, so we must process every batch independently and then aggregate
    # the results outside of the model
    # so, we create a placeholder where can feed the aggregated result back into the model
    loss_valid_pl = tf.placeholder(tf.float32, name='loss_valid_pl')
    loss_valid_s = tf.summary.scalar('loss_valid', loss_valid_pl, collections=['validation_summaries'])

    # merge validation summaries
    summaries_valid = tf.summary.merge([loss_valid_s])

    # dump the config to the model directory in case we later want to see it

    fitted_output = []
    predictions = []
    zeroPts = []
    with tf.Session() as sess:
        # Add the ops to initialize variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Actually intialize the variables
        sess.run(init_op)

        # create a saver for writing training checkpoints
        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=config['n_keep_checkpoints'])
        checkpoint = tf.train.get_checkpoint_state(config['model_dir'])
        checkpoint_restore_successful = False
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            try:
                saver.restore(sess, '%s/%s' % (os.path.join(config['model_dir']), checkpoint_name) )
                print('Checkpoint restore successful')
            except Exception as e:
                print('Checkpoint restore unsuccessful')
        # start training
        start_time = time.time()
        current_step = 0
        for e in range(config['n_epochs']):
            
            # reshuffle the batches
            #data_train.reshuffle()
            try:
                predNow = []
                # loop through all training batches
                for i, batch, labels_batch in zip(list(range(data_train.shape[0])), list(data_train), list(labels_train)):
                    step = tf.train.global_step(sess, global_step)
                    current_step += 1

                    # we want to train, so must request at least the train_op
                    fetches = {'summaries': summaries_training,
                               'loss': rnn_model.loss,
                               'train_op': train_op, 
                                'output': rnn_model.prediction}

                    feed_dict = {rnn_model.input_: batch,
                                 rnn_model.target: labels_batch}
                    # feed data into the model and run optimization
                    training_out = sess.run(fetches, feed_dict)

                    # print training performance of this batch onto console
                    time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                    if (e % 100 == 0) and i == 0:
                        print('\rEpoch: {:3d} [{:4d}/{:4d}] time: {:>8} loss: {:.4f}'.format(
                            e + 1, i + 1, 1, time_delta, training_out['loss']), end='\t')
                    
                    # save predictions
                    predNow = training_out['output']

                fitted_output=np.array(predNow[0])
            except KeyboardInterrupt:
                saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
                break
                        
            # after every 100th epoch evaluate the performance on the validation set
            total_valid_loss = 0.0
            n_valid_samples = 0
            target_labels = []
            pred_labels = []
            for batch, labels_batch in zip(list(data_cv), list(labels_cv)):
                fetches = {'loss': rnn_model_valid.loss, 
                           'output': rnn_model_valid.prediction}
                feed_dict = {rnn_model_valid.input_: batch,
                             rnn_model_valid.target: labels_batch}
                valid_out = sess.run(fetches, feed_dict)
                fitted_output = valid_out['output']
                total_valid_loss += valid_out['loss'] * batch.shape[1]
                n_valid_samples += batch.shape[1]
                target_labels.append( np.argmax(labels_batch, axis=-1))
                pred_labels.append(np.argmax(fitted_output, axis=-1))

            # write validation logs
            avg_valid_loss = total_valid_loss / n_valid_samples
            # F1 score
            target_labels = np.concatenate(target_labels, axis=0)
            pred_labels = np.concatenate(pred_labels, axis=0)
            f1_ScoreNow = f1_score(target_labels, pred_labels, average='micro')
           

            # print validation performance onto console
            if (e % 100 == 0):
                print(' | validation loss: {:.6f} | f1 score: {:.6f}'.format(avg_valid_loss, f1_ScoreNow), end='\n')

            # save this checkpoint if necessary
            if (e + 1) % config['save_checkpoints_every_epoch'] == 0:
                saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)

        # Training finished
        print('Training finished')
        ckpt_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
        print('Model saved to file {}'.format(ckpt_path))
        predictions = []
        target = []
        for batch, labels in zip(list(data_test), list(labels_test)):
            input_ = batch
            feed_dict = {rnn_model_test.input_: batch}
            fetch = [rnn_model_test.prediction]
            class_proba = sess.run(fetch, feed_dict)
            predictions.append(class_proba)
            target.append(labels)
        print('Finished evalualtion for all batches')
        predictions = np.concatenate(predictions, axis=1)
        predictions = np.argmax(predictions, axis=-1)
        predictions = np.squeeze(predictions)
        target = np.concatenate(target, axis=1)
        target = np.argmax(target, axis=-1)
        target = np.squeeze(target)
        f1_ScoreNow = f1_score(target, predictions, average='binary')
        print('f1 score of the test samples is - {:.6f}'.format(f1_ScoreNow))
        outfile = open('test_results_DNN.csv','w')
        outfile.write('ID,Prediction\n')
        for i, p in enumerate(predictions):
            outfile.write('{},{}\n'.format(i+1, p))
        outfile.close()
           

    #intgr_output = np.matmul(np.tri(fitted_output.shape[0]),fitted_output) + data_train[0,1,:]

if __name__ == '__main__':
    main(train_config)
