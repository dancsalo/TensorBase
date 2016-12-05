#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Implement Convolutional VAE for MNIST dataset to demonstrate NNClasses functionality
"""

import sys
sys.path.insert(1, '../')

from NNLayers import Layers
from NNModel import Model
from NNData import Data

import tensorflow as tf
import numpy as np
import scipy.misc
from tensorflow.examples.tutorials.mnist import input_data

# Global Dictionary of Flags
flags = {
    'save_directory': 'summaries/',
    'model_directory': 'conv_mil_vae/',
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'num_classes': 1,
    'image_dim': 28,
    'batch_size': 128,
    'display_step': 500,
    'weight_decay': 1e-4,
    'lr_decay': 0.999,
    'lr_iters': [(1e-2, 10000), (1e-3, 10000), (1e-4, 10000)]
}


class ConvMil(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.data = Mnist(flags_input)

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1], name='y')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _set_summaries(self):
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Cross Entropy Loss", self.xentropy)
        tf.scalar_summary("Weight Decay Loss", self.weight)
        tf.image_summary("x", self.x)

    def _network(self):
        with tf.variable_scope("model"):
            net = Layers(self.x)
            net.conv2d(5, 64)
            net.maxpool()
            net.conv2d(3, 64)
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(3, 128)
            net.conv2d(3, 128)
            net.conv2d(1, 64)
            net.conv2d(1, 32)
            net.conv2d(1, self.flags['num_classes'], activation_fn=tf.nn.sigmoid)
            net.noisy_and(self.flags['num_classes'])
            self.y_hat = net.get_output()

    def _optimizer(self):
        const = 1/self.flags['batch_size'] * 1/(self.flags['image_dim'] * self.flags['image_dim'])
        self.xentropy = const * tf.nn.sparse_softmax_cross_entropy_with_logits(self.y_hat, self.y, name='xentropy')
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_mean(self.xentropy + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def _generate_train_batch(self):
        self.batch_y, self.batch_x = self.data.generate_train_batch(self.flags['batch_size'])

    def _run_train_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                        feed_dict={self.x: self.batch_x, self.y: self.batch_y,
                                                   self.lr: rate})

    def _run_train_summary_iter(self):
        rate = self.learn_rate * self.flags['lr_decay']
        self.summary, self.loss, self.x_recon, _ = self.sess.run([self.merged, self.cost, self.optimizer],
                                                                 feed_dict={self.x: self.batch_x, self.y: self.batch_y,
                                                                            self.lr: rate})

    def _record_metrics(self):
        self.print_log("Batch Number: " + str(self.step) + ", Image Loss= " + "{:.6f}".format(self.loss))


class Mnist(Data):
    def __init__(self, flags):
        super().__init__(flags)

    def load_data(self):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def generate_train_batch(self, batch_size):
        batch_x, batch_y = self.mnist.train.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
        return batch_y, batch_x

    def generate_test_batch(self, batch_size):
        batch_x, batch_y = self.mnist.test.next_batch(batch_size)
        batch_x = np.reshape(batch_x, [batch_size, 28, 28, 1])
        return batch_y, batch_x


def main():
    flags['seed'] = np.random.randint(1, 1000, 1)[0]
    model_vae = ConvMil(flags, run_num=1)
    model_vae.train()

if __name__ == "__main__":
    main()