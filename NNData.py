#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Class for dataset loading and batch generation
"""

class Data:
    def __init__(self, flags):
        self.flags = flags
        self.load_data()

    def load_data(self):
        """Load the dataset into memory"""

    def generate_train_batch(self, batch_size):
        """Generate a batch of labels and images from the training set"""
        #return batch_y, batch_x

    def generate_test_batch(self, batch_size):
        """Generate a batch of labels and images from the training set"""
        #return batch_y, batch_x