#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import scipy.misc
import pickle
import sys

from ..NNClasses.NNLayers import Layers
from ..NNClasses.NNModels import Models
from record import record_metrics
from data.BREAST import BreastData


# Global Dictionary of Flags
flags = {
    'data_directory': '../../Data/',  # in relationship to the code_directory
    'previous_processed_directory': 'Smart_Crop/',
    'save_directory': 'summaries/',
    'model_directory': 'conv_vae/',
    'datasets': ['SAGE'],
    'restore': False,
    'restore_file': 'start.ckpt',
    'recon': 100000000,
    'vae': 1,
    'image_dim': 128,
    'hidden_size': 128,
    'batch_size': 16,
    'display_step': 100,
    'weight_decay': 1e-4,
    'lr_decay': 0.9999,
    'lr_iters': [(1e-3, 10000), (1e-4, 10000), (1e-5, 10000), (1e-6, 10000), (1e-7, 10000), (1e-8, 100000)]
}


class ConvVae(Models):
    def __init__(self, flags_input, model_num, image_dict):
        super().__init__(flags_input, model_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.print_log("Vae Weights: %f" % flags['vae'])
        self.print_log("Recon Weight: %d" % flags['recon'])
        self.data = BreastData(self.flags, image_dict)

    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.epsilon = tf.placeholder(tf.float32, [None, flags['hidden_size']], name='epsilon')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

    def _set_summaries(self):
        tf.scalar_summary("Total Loss", self.cost)
        tf.scalar_summary("Reconstruction Loss", self.recon)
        tf.scalar_summary("VAE Loss", self.vae)
        tf.scalar_summary("Weight Decay Loss", self.weight)
        tf.histogram_summary("Mean", self.mean)
        tf.histogram_summary("Stddev", self.stddev)
        tf.image_summary("x", self.x)
        tf.image_summary("x_hat", self.x_hat)

    def _encoder_BREAST(self, x):
        encoder = Layers(x)
        encoder.conv2d(5, 64)
        encoder.maxpool()
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 64)
        encoder.conv2d(3, 128, stride=2)
        encoder.conv2d(3, 128)
        encoder.conv2d(3, 256, stride=2)
        encoder.conv2d(3, 256)
        encoder.conv2d(3, 512, stride=2)
        encoder.conv2d(3, 512)
        encoder.conv2d(3, 1024, stride=2)
        encoder.conv2d(3, 1024)
        encoder.conv2d(1, self.flags['hidden_size'] * 2, activation_fn=None)
        encoder.avgpool(globe=True)
        return encoder.get_output()

    def _decoder_BREAST(self, z):
        if z is None:
            mean = None
            stddev = None
            input_sample = self.epsilon
        else:
            z = tf.reshape(z, [-1, self.flags['hidden_size'] * 2])
            print(z.get_shape())
            mean, stddev = tf.split(1, 2, z)
            stddev = tf.sqrt(tf.exp(stddev))
            input_sample = mean + self.epsilon * stddev
        decoder = Layers(tf.expand_dims(tf.expand_dims(input_sample, 1), 1))
        decoder.deconv2d(4, 1024, padding='VALID')
        decoder.deconv2d(3, 1024)
        decoder.deconv2d(3, 512, stride=2)
        decoder.deconv2d(3, 512)
        decoder.deconv2d(3, 256, stride=2)
        decoder.deconv2d(3, 256)
        decoder.deconv2d(3, 128, stride=2)
        decoder.deconv2d(3, 128)
        decoder.deconv2d(3, 64, stride=2)
        decoder.deconv2d(3, 64)
        decoder.deconv2d(5, 1, stride=2, activation_fn=tf.nn.tanh)
        return decoder.get_output(), mean, stddev

    def _network(self):
        with tf.variable_scope("model"):
            self.latent = self._encoder_BREAST(x=self.x)
            self.x_hat, self.mean, self.stddev = self._decoder_BREAST(z=self.latent)
        with tf.variable_scope("model", reuse=True):
            self.x_gen, _, _ = self._decoder_BREAST(z=None)

    def _optimizer(self):
        epsilon = 1e-8
        const = 1/self.flags['batch_size'] * 1/(self.flags['image_dim'] * self.flags['image_dim'])
        self.recon = const * self.flags['recon'] * tf.reduce_sum(tf.squared_difference(self.x, self.x_hat))
        self.vae = const * self.flags['vae'] * -0.5 * tf.reduce_sum(1.0 - tf.square(self.mean) - tf.square(self.stddev) + 2.0 * tf.log(self.stddev + epsilon))
        self.weight = self.flags['weight_decay'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.vae + self.recon + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def print_variable(self, var):
        self.sess.run(tf.initialize_all_variables())
        if var == 'x_recon':
            print_var = tf.Print(self.x_hat, [self.x_hat])
            norm = np.random.normal(size=[self.flags['batch_size'], self.flags['hidden_size']])
            x = np.zeros([self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])
        else:
            print('Print Variable not defined .... printing x_hat')
            return tf.Print(self.x_hat, [self.x_hat])
        return self.sess.run(print_var, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def output_shape(self):
        self.sess.run(tf.initialize_all_variables())
        norm = np.random.normal(size=[self.flags['batch_size'], self.flags['hidden_size']])
        x = np.zeros([self.flags['batch_size'], self.flags['image_dim'], self.flags['image_dim'], 1])
        return self.sess.run(self.x_hat, feed_dict={self.x: x, self.keep_prob: 0.5, self.epsilon: norm})

    def save_x_hat(self, image_generating_fxn):
        labels, x = image_generating_fxn()
        for i in range(len(x)):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(i) +'.png', np.squeeze(x[i]))
        norm = np.zeros(shape=[len(x), self.flags['hidden_size']])
        images = self.sess.run(self.x_hat, feed_dict={self.x: x, self.keep_prob: 1.0, self.epsilon: norm})
        for i in range(len(images)):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_hat' + str(i) + '.png', np.squeeze(images[i]))
        return x

    def save_x_gen(self, image_generating_fxn, num):
        labels, x = image_generating_fxn()
        print(self.flags['logging_directory'])
        for i in range(num):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(i) +'.png', np.squeeze(x[i]))
            means, stddevs = self.transform(x[0:num, :, :, :])
            print(stddevs)
            norm = np.random.normal(loc=means)
        images = self.sess.run(self.x_gen, feed_dict={self.x: x[1:num, :, :, :], self.keep_prob: 1.0, self.epsilon: norm})
        for i in range(num):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_gen' + str(i) + '.png', np.squeeze(images[i]))

    def transform(self, x):
        norm = np.random.normal(size=[x.shape[0], self.flags['hidden_size']])
        return self.sess.run([self.mean, self.epsilon], feed_dict={self.x: x, self.epsilon: norm, self.keep_prob: 1.0})

    def _generate_training_batch(self):
        self.labels_x, self.batch_x = self.data.generate_training_batch(self.global_step)
        self.norm = np.random.standard_normal([self.flags['batch_size'], self.flags['hidden_size']])

    def _run_training_iter(self):
        self.summary, _ = self.sess.run([self.merged, self.optimizer],
                                   feed_dict={self.x: self.batch_x, self.keep_prob: 1.0, self.epsilon: self.norm,
                                              self.lr: self.lr * self.flags['lr_decay']})

    def _run_training_summary_iter(self):
        self.summary, self.loss, self.x_recon, self.latent = self.sess.run(
            [self.merged, self.cost, self.x_recon, self.latent, self.optimizer],
            feed_dict={self.x: self.batch_x, self.keep_prob: 1.0, self.epsilon: self.norm, self.lr: self.lr})

    def _record_metrics(self):
        for j in range(1):
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_' + str(self.step) + '.png',
                              np.squeeze(self.batch_x[j]))
            scipy.misc.imsave(self.flags['logging_directory'] + 'x_recon_' + str(self.step) + '.png',
                              np.squeeze(self.x_recon[j]))
        record_metrics(loss=self.loss, acc=None, batch_y=None, step=self.step, split=None, flags=self.flags)
        self.print_log("Max of x: %f" % self.batch_x[1].max())
        self.print_log("Min of x: %f" % self.batch_x[1].min())
        self.print_log("Mean of x: %f" % self.batch_x[1].mean())
        self.print_log("Max of x_recon: %f" % self.x_recon[1].max())
        self.print_log("Min of x_recon: %f" % self.x_recon[1].min())
        self.print_log("Mean of x_recon: %f" % self.x_recon[1].mean())


def main():
    o = np.random.randint(1, 1000, 1)
    flags['seed'] = o[0]
    run_num = sys.argv[1]
    image_dict = pickle.load(open(flags['save_directory'] + 'preprocessed_image_dict.pickle', 'rb'))
    model_vae = ConvVae(flags, run_num, image_dict)
    model_vae.train(image_dict, model=1)
    # model.save_x(bgf)
    # x_recon = model_vae.output_shape()
    # print(x_recon.shape)
    # model_vae.restore()
    # model_vae.save_x_gen(bgf, 15)

if __name__ == "__main__":
    main()