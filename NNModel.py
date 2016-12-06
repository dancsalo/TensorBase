#!/usr/bin/env python

"""
Author: Dan Salo
Last Edit: 11/11/2016

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

import tensorflow as tf
import numpy as np
import logging
import os
import datetime


class Model:
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

    def _generate_test_batch(self):
        """Use instance of Data class to generate training batch"""

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
        flags_keys = ['restore', 'restore_file', 'batch_size','display_step', 'weight_decay', 'lr_decay', 'lr_iters']
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
