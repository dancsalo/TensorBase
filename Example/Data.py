#!/usr/bin/env python

import sys
sys.path.insert(1,'../')

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from NNData import Data


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