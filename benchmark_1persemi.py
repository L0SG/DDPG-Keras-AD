import keras.preprocessing.text as kt
import numpy as np
import random
import argparse
from keras.models import model_from_json, Model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
import tensorflow as tf
import json
import datamodule
from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import math
from keras.initializers import normal, identity, VarianceScaling
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

# Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K

K.set_session(sess)

# same preprocessing routine of the reference

# load NSL-KDD dataset
DATA_PATH = '/home/tkdrlf9202/PycharmProjects/DDPG-Keras-AD'
data_train, data_test = datamodule.loader_all(DATA_PATH)
idx_discrete = [1, 2, 3]
idx_continuous = range(len(data_train[0]))
idx_continuous = np.delete(idx_continuous, idx_discrete).tolist()

data_train_cont, data_train_disc, data_train_class = datamodule.preprocess(data_train, idx_continuous, idx_discrete)
data_test_cont, data_test_disc, data_test_class = datamodule.preprocess(data_test, idx_continuous, idx_discrete)

# generate tokenizer, one per discrete column
print('tokenizing discrete features...')
tokenizers = [kt.Tokenizer(num_words=1000) for i in xrange(len(idx_discrete))]

# tokenize discrete columns
data_train_tokenized = datamodule.generate_tokens(tokenizers, data_train_disc)
vocab_size = [min(len(tokenizers[i].word_index), tokenizers[i].num_words) for i in xrange(len(idx_discrete))]
print('vocab size of discrete features from training set : ' + str(vocab_size))
data_train_tokenized = np.array(data_train_tokenized, np.float32)

# tokenize the test set based on the training set
data_test_tokenized = [tokenizers[i].texts_to_sequences(data_test_disc[:, i]) for i in range(len(idx_discrete))]
data_test_tokenized = np.array(data_test_tokenized).reshape(data_test_cont.shape[0], -1)

# naive concat for tokinized values
# needs embedding
data_train_x = np.concatenate((data_train_cont, data_train_tokenized), axis=1)
data_train_x = normalize(data_train_x, axis=0)

data_test_x = np.concatenate((data_test_cont, data_test_tokenized), axis=1)
data_test_x = normalize(data_test_x, axis=0)

data_train_y = []
for elem in data_train_class:
    if elem == 'normal':
        data_train_y.append(0.)
    else:
        data_train_y.append(1.)

data_test_y = []
for elem in data_test_class:
    if elem == 'normal':
        data_test_y.append(0.)
    else:
        data_test_y.append(1.)

unique, counts = np.unique(data_train_y, return_counts=True)
print(dict(zip(unique, counts)))
unique, counts = np.unique(data_test_y, return_counts=True)
print(dict(zip(unique, counts)))

# split the train data for validation
data_train_x, data_valid_x, data_train_y, data_valid_y = train_test_split(data_train_x,
                                                                          data_train_y,
                                                                          test_size=0.2)

#############################################

action_dim = 1  # anomaly
state_dim = 41  # input dimension
HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
LRA = 0.0001


# FF network with the same size of the actor network

class benchmark_FF():
    def __init__(self):
        S = Input(shape=[state_dim])
        h0 = Dense(HIDDEN1_UNITS, activation='relu')(S)
        h1 = Dense(HIDDEN2_UNITS, activation='relu')(h0)
        V = Dense(1, activation='sigmoid')(h1)
        model = Model(inputs=S, outputs=V)
        model.compile(Adam(LRA), loss='mse', metrics=['accuracy'])
        self.model = model


bench_ff = benchmark_FF()

log_file = open('taining_log_bench_33persemi.txt', 'w+')
for i in range(100):
    # shuffle the dataset
    feed = list(zip(data_train_x, data_train_y))
    random.shuffle(feed)
    data_train_x, data_train_y = zip(*feed)
    data_train_x = np.array(data_train_x)
    data_train_y = np.array(data_train_y)

    accuracy_train = []
    for j in range(data_train_x.shape[0]):
        if i == 0:
            if j >= 9999:
                # train with ground truth one per 100 steps
                if j % 3 == 0:
                    acc = bench_ff.model.train_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]),
                                                  np.expand_dims(data_train_y[j], axis=0))
                    accuracy_train.append(np.array(acc[1]))
                # train with pseudo-label for 99 steps
                else:
                    pred = bench_ff.model.predict_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]))
                    pseudo_label = np.around(pred[0][0])
                    acc = np.where(pseudo_label == data_train_y[j], 1., 0.)
                    accuracy_train.append(acc)
                    bench_ff.model.train_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]),
                                                  np.expand_dims(pseudo_label, axis=0))
            else:
                acc = bench_ff.model.train_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]),
                                          np.expand_dims(data_train_y[j], axis=0))
                accuracy_train.append(np.array(acc[1]))
        else:
            # train with ground truth one per 100 steps
            if j % 3 == 0:
                acc = bench_ff.model.train_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]),
                                              np.expand_dims(data_train_y[j], axis=0))
                accuracy_train.append(np.array(acc[1]))
            # train with pseudo-label for 99 steps
            else:
                pred = bench_ff.model.predict_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]))
                pseudo_label = np.around(pred[0][0])
                acc = np.where(pseudo_label == data_train_y[j], 1., 0.)
                accuracy_train.append(acc)
                bench_ff.model.train_on_batch(data_train_x[j].reshape(1, data_train_x[j].shape[0]),
                                              np.expand_dims(pseudo_label, axis=0))

    accuracy_train = np.mean(np.array(accuracy_train))

    preds_val = []
    for j in range(data_valid_x.shape[0]):
        preds_val.append(bench_ff.model.predict(data_valid_x[j].reshape(1, data_valid_x[j].shape[0]))[0][0])
    preds_val_ = np.where(np.array(preds_val) >= 0.5, 1, 0)
    accuracy_val = accuracy_score(data_valid_y, preds_val_)
    print("epoch", i, ": train accuracy", accuracy_train, ", validation accuracy ", accuracy_val)

    # epoch (equiv. as episode) / training accuracy / validation accuracy
    log_string = str(i) + '\t' + \
                 str(accuracy_train) + '\t' + \
                 str(accuracy_val) + '\n'
    log_file.write(log_string)

