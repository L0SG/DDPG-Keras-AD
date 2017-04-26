import tensorflow as tf
import keras
import numpy as np
import os
import keras.backend as K
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from keras.utils import to_categorical
import csv

def loader_all(data_path):
    """
    load the dataset from the original path
    assumes separate train, valid, and train directory locations
    :param data_path: the master math
    :return: train, valid, and test set of ndarray format
    """

    # define the path
    path_train = os.path.join(data_path, 'train')
    path_test = os.path.join(data_path, 'test')

    data_train = []
    data_test = []

    for f in glob.glob(os.path.join(path_train, "*.csv")):
        csv_file = open(f, 'rU')
        csv_data = csv.reader(csv_file)
        data_train = list(csv_data)

    for f in glob.glob(os.path.join(path_test, "*.csv")):
        csv_file = open(f, 'rU')
        csv_data = csv.reader(csv_file)
        data_test = list(csv_data)

    data_train = np.array(data_train)
    data_test = np.array(data_test)

    return data_train, data_test


def loader_separate(data_path):
    """
    load the dataset from the original path
    uses append instaed of extend: separate subarray per file
    :param data_path: the master math
    :return: dataset of list of ndarray format
    """
    data = []

    for f in glob.glob(os.path.join(data_path, "*.txt")):
        lines = open(f, 'r').readlines()
        data.append(np.asarray([line.strip().split() for line in lines]))

    data = np.asarray(data)

    return data


def preprocess(data, idx_continuous, idx_discrete):
    """
    preprocess the data, continuous & discrete
    :param data: raw data in np matrix form
    :return: data_continuous, data_discrete
    """
    # load discrete & continuous columns separately
    data_discrete = data[:, idx_discrete]
    data_continuous = data[:, idx_continuous]
    """
    # TEMPORARY HACK: REMOVE DOTS IN IP ADDRESS
    # dots IN THE IP make the tokenizer API split the IP address to 4 elements
    # so just remove them
    data_discrete[:, IDX_IP] = np.char.replace(data_discrete[:, IDX_IP], '.', '')
    """
    IDX_SERVICE = 1
    # ANOTHER HACK: REMOVE _ IN REGION NAME
    # tokenizer gets multiple words, regarding _ as separator
    data_discrete[:, IDX_SERVICE] = np.char.replace(data_discrete[:, IDX_SERVICE], '_', '')


    data_class = data_continuous[:, -2]
    data_continuous = data_continuous[:, :-2]

    # cast to float and normalize continuous features per column
    # storing the mean is necessary for the valid and test set
    # it'd be better to normalize the data prior to the code
    data_continuous = data_continuous.astype('float32')
    """
    data_continuous = np.log(np.add(data_continuous, 1))
    
    scaler = [preprocessing.MinMaxScaler(feature_range=(-0.5, 0.5))for i in xrange(len(idx_continuous))]
    for i in xrange(len(idx_continuous)):
        scaler[i].fit(data_continuous[i].reshape(-1, 1))
    """
    return data_continuous, data_discrete, data_class


def preprocess_feed(data, idx_continuous, idx_discrete, IDX_IP, IDX_REGION):
    """
    preprocess the data, continuous & discrete without scaler
    :param data: raw data in np matrix form
    :return: data_continuous, data_discrete
    """
    # load discrete & continuous columns separately
    data_discrete = data[:, idx_discrete]
    data_continuous = data[:, idx_continuous]

    # TEMPORARY HACK: REMOVE DOTS IN IP ADDRESS
    # dots IN THE IP make the tokenizer API split the IP address to 4 elements
    # so just remove them
    data_discrete[:, IDX_IP] = np.char.replace(data_discrete[:, IDX_IP], '.', '')

    # ANOTHER HACK: REMOVE - IN REGION NAME
    # tokenizer gets multiple words, regarding - as separator
    data_discrete[:, IDX_REGION] = np.char.replace(data_discrete[:, IDX_REGION], '-', '')

    # cast to float and normalize continuous features per column
    # storing the mean is necessary for the valid and test set
    # it'd be better to normalize the data prior to the code
    data_continuous = data_continuous.astype('float32')
    data_continuous = np.log(np.add(data_continuous, 1))

    return data_continuous, data_discrete


def generate_tokens(tokenizers, data_discrete):
    """
    generate token matrix, with one tokenizer per feature column
    :param tokenizers: list of tokenizer objects
    :param data_discrete: original data containing discrete features
    :return: tokenized matrix data_tokenized
    """

    # initialize tokenized matrix
    # tokens must be int
    data_tokenized = np.zeros((len(data_discrete), len(data_discrete[0])), dtype='int32')

    # tokenizing loop
    # CAUTION: dots (.) in IP address feature cause fit_on_texts split the IP string to 4 elements
    # do not directly feed the IP address data to this loop
    for i in xrange(len(data_discrete[0])):
        tokenizers[i].fit_on_texts(data_discrete[:, i])
        tokens = tokenizers[i].texts_to_sequences(data_discrete[:, i])
        # handle the unknown token suppressed by max vocab size
        # currently empty element, need to supplement for 0
        tokens = [element or [0] for element in tokens]
        tokens = np.array(tokens)
        tokens = tokens.reshape((1, -1))
        data_tokenized[:, i] = tokens

    return data_tokenized


def make_feed(data_path, tokenizers, scaler, idx_continuous, idx_discrete, batch_size, IDX_IP, IDX_REGION):
    # formulate x and y for model.fit, corresponding to the defined input above

    # load multiple files separately as list elements
    data = loader_separate(data_path)

    # preprocess the data similar to the previous method
    data_continuous = []
    data_tokenized = []
    # preprocess per files
    for i in xrange(len(data)):
        cont, disc = preprocess_feed(data[i], idx_continuous, idx_discrete, IDX_IP, IDX_REGION)
        tokenized = np.zeros((len(disc), len(disc[0])), dtype='int32')
        # tokenize based on the trained tokenizers
        for j in xrange(len(disc[0])):
            tokens = tokenizers[j].texts_to_sequences(disc[:, j])
            tokens = [element or [0] for element in tokens]
            tokens = np.array(tokens)
            tokens = tokens.reshape((1, -1))
            tokenized[:, j] = tokens
        data_continuous.append(cont)
        data_tokenized.append(tokenized)
    assert len(data_continuous) == len(data_tokenized)

    feed_x_list = []
    feed_y_list = []
    for idx in xrange(len(data_continuous)):
        feed_x_discrete = [data_tokenized[idx][:-1, i] for i in xrange(len(idx_discrete))]
        feed_x_continuous = [data_continuous[idx][:-1, i] for i in xrange(len(idx_continuous))]

        # if the file has just one sample, skip it (training not possible)
        if feed_x_continuous[0].shape[0] == 0:
            continue

        # scale continuous features per column
        for i in xrange(len(idx_continuous)):
            feed_x_continuous[i] = scaler[i].transform(feed_x_continuous[i].reshape(-1, 1)).reshape(-1)
        """
        batch_residual_zero_flag = False
        if feed_x_discrete[0].shape[0] % batch_size == 0:
            batch_residual_zero_flag = True

        # ditch the last several samples not matching the batch size
        if batch_residual_zero_flag == False:
            feed_x_discrete = [feed_x_discrete[i][:-(feed_x_discrete[i].shape[0] % batch_size)] for i in
        xrange(len(feed_x_discrete))]
            feed_x_continuous = [feed_x_continuous[0][:-(feed_x_continuous[0].shape[0] % batch_size), :]]
        else:
            feed_x_discrete = feed_x_discrete
            feed_x_continuous = [feed_x_continuous[0]]
        """
        # merge
        feed_x = feed_x_discrete + feed_x_continuous

        # same goes to y, but should convert to one hot encoding for discrete features
        feed_y_discrete = [data_tokenized[idx][1:, i] for i in xrange(len(idx_discrete))]
        feed_y_continuous = [data_continuous[idx][1:, i] for i in xrange(len(idx_continuous))]
        for i in xrange(len(idx_continuous)):
            feed_y_continuous[i] = scaler[i].transform(feed_y_continuous[i].reshape(-1, 1)).reshape(-1)
        """
        if batch_residual_zero_flag == False:
            feed_y_discrete = [feed_y_discrete[i][:-(feed_y_discrete[i].shape[0] % batch_size)] for i in
                               xrange(len(feed_y_discrete))]
            feed_y_continuous = [feed_y_continuous[0][:-(feed_y_continuous[0].shape[0] % batch_size), :]]
        else:
            feed_y_discrete = feed_y_discrete
            feed_y_continuous = [feed_y_continuous[0]]
        """
        """
        # to_categorical makes preprocessing speed bottleneck
        # needs fix
        feed_y_categorical = []
        for i in xrange(len(idx_discrete)):
            # vocabulary is one-based, and to_categorical is zero-based (fuck)
            # feed_y_discrete[i] = np.add(feed_y_discrete[i], -1)
            # +1 of classes for unknown token (denoted by 0)
            feed_y_categorical.append(to_categorical(feed_y_discrete[i],
                                                     num_classes=min(len(tokenizers[i].word_index), tokenizers[i].num_words)+1))
        feed_y = feed_y_categorical + feed_y_continuous
        """
        feed_y = feed_y_discrete + feed_y_continuous

        feed_x_list.append(feed_x)
        feed_y_list.append(feed_y)

    return feed_x_list, feed_y_list


def expand_dims(x):
    return K.expand_dims(x, 1)


def expand_dims_output_shape(input_shape):
    return (input_shape[0], 1, input_shape[1])