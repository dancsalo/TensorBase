#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Class for dataset loading and batch generation
"""

import numpy as np
import math

class Data:
    def __init__(self, flags, valid_percent=0.2, test_percent=0.15):
        self.flags = flags
        train_images, train_labels, self.test_images, self.test_labels = self.load_data(test_percent)
        self._num_test_images = len(self.test_labels)
        self._num_train_images = math.floor(len(train_labels) * (1 - valid_percent))
        self._num_valid_images = len(train_labels) - self._num_train_images
        self.train_images, self.train_labels, self.valid_images, self.valid_labels =\
            self.split_data(train_images, train_labels)

        self.train_epochs_completed = 0
        self.index_in_train_epoch = 0
        self.index_in_valid_epoch = 0
        self.index_in_test_epoch = 0

    def load_data(self, test_percent):
        """Load the dataset into memory. If data is not divded into train/test, use test_percent"""
        # return train_images, train_labels, test_images, test_labels'''

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
