#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 11/11/2016

Purpose: Class for convolutional model creation similar to Keras with layer-by-layer formulation.
Example:
    x ### is a numpy 4D array
    encoder = Layers(input)
    encoder.conv2d(3, 64)
    encoder.conv2d(3, 64)
    encoder.maxpool()
    ...
    encoder.get_output()
    ...
    decoder = Layers(z)
    decoder.deconv2d(4, 156, padding='VALID')
    decoder.deconv2d(3, 144, stride=2)
    decoder.deconv2d(5, 128, stride=2)
    ...
"""

import tensorflow as tf
import numpy as np
import logging
import tensorflow.contrib.layers as init
import math
import os
import datetime


class Layers:
    def __init__(self, x):
        """
        Initialize model Layers.
        .input = numpy array
        .count = dictionary to keep count of number of certain types of layers for naming purposes
        """
        self.input = x
        self.count = {'conv': 1, 'deconv': 1, 'fc': 1, 'flat': 1, 'mp': 1, 'up': 1, 'ap': 1}

    def conv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0):
        """
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        scope = 'conv_' + str(self.count['conv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, input_channels, output_channels]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.nn.conv2d(self.input, w, strides=[1, stride, stride, 1], padding=padding)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = self.batch_norm(self.input, s)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        
        self.count['conv'] += 1

    def deconv2d(self, filter_size, output_channels, stride=1, padding='SAME', activation_fn=tf.nn.relu, b_value=0.0, s_value=1.0):
        """
        :param filter_size: int. assumes square filter
        :param output_channels: int
        :param stride: int
        :param padding: 'VALID' or 'SAME'
        :param activation_fn: tf.nn function
        :param b_value: float
        :param s_value: float
        """
        scope = 'deconv_' + str(self.count['deconv'])
        with tf.variable_scope(scope):
            input_channels = self.input.get_shape()[3]
            output_shape = [filter_size, filter_size, output_channels, input_channels]
            w = self.weight_variable(name='weights', shape=output_shape)

            batch_size = tf.shape(self.input)[0]
            input_height = tf.shape(self.input)[1]
            input_width = tf.shape(self.input)[2]
            filter_height = tf.shape(w)[0]
            filter_width = tf.shape(w)[1]
            out_channels = tf.shape(w)[2]
            row_stride = stride
            col_stride = stride

            if padding == "VALID":
                out_rows = (input_height - 1) * row_stride + filter_height
                out_cols = (input_width - 1) * col_stride + filter_width
            else:  # padding == "SAME":
                out_rows = input_height * row_stride
                out_cols = input_width * col_stride

            out_shape = tf.pack([batch_size, out_rows, out_cols, out_channels])

            self.input = tf.nn.conv2d_transpose(self.input, w, out_shape, [1, stride, stride, 1], padding)
            if s_value is not None:
                s = self.const_variable(name='scale', shape=[output_channels], value=s_value)
                self.input = self.batch_norm(self.input, s)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_channels], value=b_value)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['deconv'] += 1

    def flatten(self, keep_prob=1):
        """
        :param keep_prob: int. set to 1 for no dropout
        """
        scope = 'flat_' + str(self.count['flat'])
        with tf.variable_scope(scope):
            input_nodes = tf.Dimension(self.input.get_shape()[1] * self.input.get_shape()[2] * self.input.get_shape()[3])
            output_shape = tf.pack([-1, input_nodes])
            self.input = tf.reshape(self.input, output_shape)
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['flat'] += 1

    def fc(self, output_nodes, keep_prob=1, activation_fn=tf.nn.relu, b_value=0.0):
        """
        :param output_nodes: int
        :param keep_prob: int. set to 1 for no dropout
        :param activation_fn: tf.nn function
        :param b_value: float or None
        """
        scope = 'fc_' + str(self.count['fc'])
        with tf.variable_scope(scope):
            input_nodes = self.input.get_shape()[1]
            output_shape = [input_nodes, output_nodes]
            w = self.weight_variable(name='weights', shape=output_shape)
            self.input = tf.matmul(self.input, w)
            if b_value is not None:
                b = self.const_variable(name='bias', shape=[output_nodes], value=0.0)
                self.input = tf.add(self.input, b)
            if activation_fn is not None:
                self.input = activation_fn(self.input)
            if keep_prob != 1:
                self.input = tf.nn.dropout(self.input, keep_prob=keep_prob)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['fc'] += 1

    def unpool(self, k=2):
        """
        :param k: int
        """
        # Source: https://github.com/tensorflow/tensorflow/issues/2169
        # Not Yet Tested
        bottom_shape = tf.shape(self.input)
        top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

        batch_size = top_shape[0]
        height = top_shape[1]
        width = top_shape[2]
        channels = top_shape[3]

        argmax_shape = tf.to_int64([batch_size, height, width, channels])
        output_list = [k // (argmax_shape[2] * argmax_shape[3]),
                       k % (argmax_shape[2] * argmax_shape[3]) // argmax_shape[3]]
        argmax = tf.pack(output_list)

        t1 = tf.to_int64(tf.range(channels))
        t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
        t1 = tf.reshape(t1, [-1, channels])
        t1 = tf.transpose(t1, perm=[1, 0])
        t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
        t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

        t2 = tf.to_int64(tf.range(batch_size))
        t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
        t2 = tf.reshape(t2, [-1, batch_size])
        t2 = tf.transpose(t2, perm=[1, 0])
        t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

        t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

        t = tf.concat(4, [t2, t3, t1])
        indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

        x1 = tf.transpose(self.input, perm=[0, 3, 1, 2])
        values = tf.reshape(x1, [-1])

        delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
        return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))

    def maxpool(self, k=2, globe=False):
        """
        :param k: int
        :param globe:  int, whether to pool over each feature map in its entirety
        """
        scope = 'maxpool_' + str(self.count['mp'])
        if globe is True:  # self.input must be a 4D image stack
            k1 = self.input.get_shape()[1]
            k2 = self.input.get_shape()[2]
            s1 = 1
            s2 = 1
            padding = 'VALID'
        else:
            k1 = k
            k2 = k
            s1 = k
            s2 = k
            padding = 'SAME'
        with tf.variable_scope(scope):
            self.input = tf.nn.max_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['mp'] += 1

    def avgpool(self, k=2, globe=False):
        """
         :param k: int
         :param globe: int, whether to pool over each feature map in its entirety
         """
        scope = 'avgpool_' + str(self.count['mp'])
        if globe is True:  # self.input must be a 4D image stack
            k1 = self.input.get_shape()[1]
            k2 = self.input.get_shape()[2]
            s1 = 1
            s2 = 1
            padding = 'VALID'
        else:
            k1 = k
            k2 = k
            s1 = k
            s2 = k
            padding = 'SAME'
        with tf.variable_scope(scope):
            self.input = tf.nn.avg_pool(self.input, ksize=[1, k1, k2, 1], strides=[1, s1, s2, 1], padding=padding)
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))
        self.count['ap'] += 1

    def noisy_and(self, num_classes):
        scope = 'noisyAND'
        with tf.variable_scope(scope):
            a = self.const_variable(name='a', shape=[1], value=1.0)
            b = self.const_variable(name='b', shape=[1, num_classes], value=0.0)
            mean = tf.reduce_mean(self.input, axis=[1, 2])
            self.input = (tf.nn.sigmoid(a*(mean-b))-tf.nn.sigmoid(-a*b))/(tf.sigmoid(a*(1-b))-tf.sigmoid(-a*b))
        self.print_log(scope + ' output: ' + str(self.input.get_shape()))

    def weight_variable(self, name, shape):
        """
        :param name: string
        :param shape: 4D array
        :return: tf variable
        """
        w = tf.get_variable(name=name, shape=shape, initializer=init.variance_scaling_initializer())
        weights_norm = tf.reduce_sum(tf.nn.l2_loss(w), name=name + '_norm')
        tf.add_to_collection('weight_losses', weights_norm)
        return w

    def get_output(self):
        """
        call at the last layer of the network.
        """
        return self.input

    @staticmethod
    def batch_norm(x, s, epsilon=1e-3):
        """
        :param x: input feature map stack
        :param s: constant tf variable
        :param epsilon: float
        :return: output feature map stack
        """
        # Calculate batch mean and variance
        batch_mean1, batch_var1 = tf.nn.moments(x, [0], keep_dims=True)

        # Apply the initial batch normalizing transform
        z1_hat = (x - batch_mean1) / tf.sqrt(batch_var1 + epsilon)
        z1_hat = z1_hat * s
        return z1_hat

    @staticmethod
    def print_log(message):
        print(message)
        logging.info(message)

    @staticmethod
    def const_variable(name, shape, value):
        """
        :param name: string
        :param shape: 1D array
        :param value: float
        :return: tf variable
        """
        return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


class Data:
    def __init__(self, flags, valid_percent=0.2, test_percent=0.15):
        self.flags = flags
        train_images, train_labels, self.test_images, self.test_labels = self.load_data(test_percent)
        self._num_test_images = len(self.test_labels)
        self._num_train_images = math.floor(len(train_labels) * (1 - valid_percent))
        self._num_valid_images = len(train_labels) - self._num_train_images
        self.train_images, self.train_labels, self.valid_images, self.valid_labels = \
            self.split_data(train_images, train_labels)

        self.train_epochs_completed = 0
        self.index_in_train_epoch = 0
        self.index_in_valid_epoch = 0
        self.index_in_test_epoch = 0

    def load_data(self, test_percent=0.15):
        """Load the dataset into memory. If data is not divded into train/test, use test_percent"""
        train_images = list()
        train_labels = list()
        test_images = list()
        test_labels = list()
        return train_images, train_labels, test_images, test_labels

    def split_data(self, train_images, train_labels):
        """
        :param train_images: numpy array (image_dim, image_dim, num_images)
        :param train_labels: numpy array (
        :return: train_images, train_labels, valid_images, valid_labels
        """
        valid_images = train_images[:self.num_valid_images]
        valid_labels = train_labels[:self.num_valid_images]
        train_images = train_images[self.num_valid_images:]
        train_labels = train_labels[self.num_valid_images:]
        return train_images, train_labels, valid_images, valid_labels

    def next_train_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self.index_in_train_epoch
        self.index_in_train_epoch += batch_size
        if self.index_in_train_epoch > self.num_train_images:
            # Finished epoch
            self.train_epochs_completed += 1

            # Shuffle the data
            perm = np.arange(self.num_train_images)
            np.random.shuffle(perm)
            self.train_images = self.train_images[perm]
            self.train_labels = self.train_labels[perm]

            # Start next epoch
            start = 0
            self.index_in_train_epoch = batch_size
            assert batch_size <= self.num_train_images

        end = self.index_in_train_epoch
        return self.train_labels[start:end], self.train_images[start:end]

    def next_valid_batch(self, batch_size):
        start = self.index_in_valid_epoch
        if self.index_in_valid_epoch + batch_size > self.num_valid_images:
            batch_size = 1
        self.index_in_valid_epoch += batch_size
        end = self.index_in_valid_epoch
        return self.valid_labels[start:end], self.valid_images[start:end], end, batch_size

    def next_test_batch(self, batch_size):
        start = self.index_in_test_epoch
        print(start)
        if self.index_in_test_epoch + batch_size > self.num_test_images:
            batch_size = 1
        self.index_in_test_epoch += batch_size
        end = self.index_in_test_epoch
        return self.test_labels[start:end], self.test_images[start:end], end, batch_size

    @property
    def num_train_images(self):
        return self._num_train_images

    @property
    def num_test_images(self):
        return self._num_test_images

    @property
    def num_valid_images(self):
        return self._num_valid_images


class Model:
    """
    Author: Dan Salo
    Initial Commit: 11/11/2016

    Purpose: Parent Class for all models creation
    Example:
        x ### is a numpy 4D array
        encoder = Layers(input)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.maxpool()
        ...
        decoder = Layers(z)
        decoder.deconv2d(4, 156, padding='VALID')
        decoder.deconv2d(3, 144, stride=2)
        decoder.deconv2d(5, 128, stride=2)
        ...
    """
    def __init__(self, flags, run_num):
        print(flags)
        self.run_num = run_num
        self.flags = self._check_flags(flags)
        self._check_file_io(run_num)
        self._set_placeholders()
        self._set_seed()
        self._define_data()

        self._network()
        self._optimizer()

        self._set_summaries()
        self.merged, self.saver, self.sess, self.writer = self._set_tf_functions()
        self._initialize_model()
        self.global_step = 1

    def _set_placeholders(self):
        """Define placeholder"""

    def _define_data(self):
        """Define data class object """
        self.num_test_images = 0
        self.num_valid_images = 0
        self.num_train_images = 0

    def _network(self):
        """Define network"""

    def _optimizer(self):
        """Define optimizer"""

    def _generate_train_batch(self):
        """Use instance of Data class to generate training batch"""

    def _generate_valid_batch(self):
        """Use instance of Data class to generate validation batch"""
        valid_number = 0
        return valid_number

    def _generate_test_batch(self):
        """Use instance of Data class to generate training batch"""
        test_number = 0
        return test_number

    def _run_train_iter(self):
        """run sess.run on optimizer"""

    def _run_valid_iter(self):
        """run sess.run on labels only to computer accuracy on validation set"""

    def _run_test_iter(self):
        """run sess.run on labels only to computer accuracy on test set"""

    def _run_train_summary_iter(self):
        """run sess.run on optimizer and merged summaries"""
        self.summary = 'object to be defined'

    def _record_test_metrics(self):
        """Define and save metrics for testing"""

    def _record_valid_metrics(self):
        """Define and save metrics for validation"""

    def _record_train_metrics(self):
        """Define and save metrics for training"""

    def _check_file_io(self, run_num):
        folder = 'Model' + str(run_num) + '/'
        self.flags['restore_directory'] = self.flags['save_directory'] + self.flags['model_directory'] + folder
        self.make_directory(self.flags['restore_directory'])
        logging.basicConfig(filename=self.flags['restore_directory'] + 'ModelInformation.log', level=logging.INFO)

    def _set_seed(self):
        tf.set_random_seed(self.flags['seed'])
        np.random.seed(self.flags['seed'])

    def _set_summaries(self):
        for var in tf.trainable_variables():
            tf.histogram_summary(var.name, var)

    def _set_tf_functions(self):
        merged = tf.merge_all_summaries()
        saver = tf.train.Saver()
        sess = tf.InteractiveSession()
        writer = tf.train.SummaryWriter(self.flags['restore_directory'], sess.graph)
        return merged, saver, sess, writer

    def _restore(self):
        new_saver = tf.train.import_meta_graph('./' + self.flags['restore_directory'] + self.flags['restore_file'])
        new_saver.restore(self.sess, tf.train.latest_checkpoint('./' + self.flags['restore_directory']))
        self.print_log("Model restored from %s" % self.flags['restore_file'])

    def _setup_metrics(self):
        self.print_log('Date: ' + str(datetime.datetime.now()).split('.')[0])
        datasets = 'Datasets: '
        for d in self.flags['datasets']:
            datasets += d + ', '
        self.print_log(datasets)
        self.print_log('Batch_size: ' + self.check_str(self.flags['batch_size']))
        self.print_log('Model: ' + self.check_str(self.flags['model_directory']))
        for l in range(len(self.flags['lr_iters'])):
            self.print_log('EPOCH %d' % l)
            self.print_log('Learning Rate: ' + str(self.flags['lr_iters'][l][0]))
            self.print_log('Iterations: ' + str(self.flags['lr_iters'][l][1]))

    def _initialize_model(self):
        self._setup_metrics()
        if self.flags['restore'] is True:
            self._restore()
        else:
            self.sess.run(tf.global_variables_initializer())
            self.print_log("Model training from scratch.")

    def _save_model(self, section):
        self.print_log("Optimization Finished!")
        checkpoint_name = self.flags['restore_directory'] + 'part_%d' % section + '.ckpt'
        save_path = self.saver.save(self.sess, checkpoint_name)
        self.print_log("Model saved in file: %s" % save_path)

    def _record_training_step(self):
        self.writer.add_summary(summary=self.summary, global_step=self.global_step)
        self.step += 1
        self.global_step += 1

    def train(self):
        for i in range(len(self.flags['lr_iters'])):
            self.step = 1
            self.learn_rate = self.flags['lr_iters'][i][0]
            self.iters_num = self.flags['lr_iters'][i][1]
            self.print_log('Learning Rate: %d' % self.learn_rate)
            self.print_log('Iterations: %d' % self.iters_num)
            while self.step < self.iters_num:
                print('Batch number: %d' % self.step)
                self._generate_train_batch()
                if self.step % self.flags['display_step'] != 0:
                    self._run_train_iter()
                else:
                    self._run_train_summary_iter()
                    self._record_train_metrics()
                self._record_training_step()
            self._save_model(section=i)

    def valid(self):
        self.print_log('Begin validation sequence')
        valid_number = 0
        while valid_number < self.num_valid_images:
            valid_number = self._generate_valid_batch()
            self._run_valid_iter()
            print(valid_number)
        self._record_valid_metrics()

    def test(self):
        self.print_log('Begin test sequence')
        test_number = 0
        while test_number < self.num_test_images:
            test_number = self._generate_test_batch()
            self._run_test_iter()
            print(test_number)
        self._record_test_metrics()

    @staticmethod
    def _check_flags(flags):
        flags_keys = ['restore', 'restore_file', 'batch_size', 'display_step', 'weight_decay', 'lr_decay',
                      'lr_iters']
        for k in flags_keys:
            try:
                flags[k]
            except KeyError:
                print('The key %s is not defined in the flags dictionary. Please define and run again' % k)
        return flags

    @staticmethod
    def make_directory(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    @staticmethod
    def print_log(message):
        print(message)
        logging.info(message)

    @staticmethod
    def check_str(obj):
        if isinstance(obj, str):
            return obj
        if isinstance(obj, float):
            return str(int(obj))
        else:
            return str(obj)
