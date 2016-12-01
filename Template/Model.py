#!/usr/bin/env python

"""
Author: Dan Salo
Initial Commit: 12/1/2016

Purpose: Template for Model Class in Tensorflow

GLOBAL DICTIONARY OF FLAGS:
The following flags must be defined:
'save_directory': string. folder name containing all models.
'model_directory': string. folder name of model which contains image printouts and model checkpoints.
'restore': boolean. Whether to restore from restore_file
'restore_file': string. Filename of checkpoint file, assumed to be in save_directory.
'image_dim': int. Assumes square images.
'hidden_size': int.
'batch_size': int.
'display_step': int.
'weight_decay': float.
'seed': int.
'lr_decay': float.
'lr_iters': list of length two tuples. First element is learning rate, second element in number of iterations.
"""

import sys
sys.path.insert(1, '../')

from NNModel import Model
from NNLayers import Layers
from NNData import Data


flags = {}


class TemplateModel(Model):
    def __init__(self, flags_input, run_num):
        """Call parent init function and define Data class object"""
        super().__init__(flags_input, run_num)
        self.data = Template()

    def _set_placeholders(self):
        """Define all placeholder variables that will be fed into feed dictionary of sess.run call"""

    def _set_summaries(self):
        """Define all placeholder variables that will be fed into feed dictionary of sess.run call"""

    def _network(self):
        """Define network, preferably using Layers class object
        How to use Layers Class:
        x  ### is a numpy 4D array
        encoder = Layers(input)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.maxpool()
        ...
        encoder.get_output()  # returns numpy array
        """
        self.nn = Layers(x=[])

    def _optimizer(self):
        """Define tensorflow optimizer"""

    def _generate_train_batch(self):
        """Use Data object class to generate training batch"""

    def _run_train_iter(self):
        """run sess.run on optimizer and merged (for tensorboard summaries)"""

    def _run_train_summary_iter(self):
        """run sess.run on optimizer and also cost/loss"""
        self.summary = 'object to be defined'

    def _record_metrics(self):
        """Define and save metrics"""


class TemplateData(Data):
    def __init__(self, flags):
        """Call parent init function"""
        super().__init__(flags)

    def load_data(self):
        """Load the dataset into memory"""

    def generate_train_batch(self, batch_size):
        """Generate a batch of labels and images from the training set"""
        # This function must return batch_y, batch_x

    def generate_test_batch(self, batch_size):
        """Generate a batch of labels and images from the training set"""
        # This function must return batch_y, batch_x


def main():
    model = TemplateModel(flags, run_num=1)
    model.train()

if __name__ == "__main__":
    main()