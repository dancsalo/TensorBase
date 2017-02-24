from .base import Layers
import tensorflow as tf
import math


class StochLayer:
    def __init__(self, x, num_latent, eq_samples, iw_samples):
        self.x = x
        self.batch_size = self.x.get_shape()[0]
        self.num_latent = num_latent
        self.params = self.compute_params
        self.samples = self.get_samples
        self.iw_samples = iw_samples
        self.eq_samples = eq_samples

    def compute_params(self):
        raise NotImplementedError

    @property
    def get_params(self):
        return self.params

    @property
    def get_samples(self):
        return self.samples


class GaussianLayerFC(StochLayer):
    def __init__(self, x, num_latent, eq_samples=1, iw_samples=1):
        super().__init__(x, num_latent, eq_samples, iw_samples)
        self.mu, self.std = self.params

    def compute_params(self):

        # Infer mu and std with fully connected layer
        model_mu = Layers(self.x)
        model_mu.fc(self.num_latent)
        mu = model_mu.get_output()
        model_var = Layers(self.x)
        model_var.fc(self.num_latent)
        std = tf.nn.softplus(model_var.get_output())

        # Make mu and std 4D for eq_samples and iw_samples
        std = tf.expand_dims(tf.expand_dims(std, 1), 1)
        mu = tf.expand_dims(tf.expand_dims(mu, 1), 1)
        return mu, std

    def compute_samples(self):
        """ Sample from a Normal distribution with inferred mu and std """
        eps = tf.random_normal([self.batch_size, self.eq_samples, self.iw_samples, self.num_latent])
        z = tf.reshape(eps * self.std + self.mu, [-1, self.num_latent])
        return z

    def neg_log_likelihood(self, x, standard=False):
        """ Calculate Log Likelihood with particular mean and std
        x must be 2D. [batch_size * eqsamples* iwsamples, num_latent]
        """
        x_reshape = tf.reshape(x, [self.batch_size, self.eq_samples, self.iw_samples, self.num_latent])
        c = - 0.5 * math.log(2 * math.pi)
        if standard is False:
            density = c - tf.log(self.std) - (x_reshape - self.mu) ** 2 / (2 * self.std**2)
        else:
            density = c - (x_reshape - self.mu) ** 2 / 2
        # sum over all importance weights. average over all eq_samples
        return tf.reduce_mean(tf.reduce_sum(-density, axis=2), axis=(1, 2))


class GaussianLayerConv(StochLayer):
    def __init__(self, x, num_latent, eq_samples=1, iw_samples=1):
        super().__init__(x, num_latent, eq_samples, iw_samples)
        self.mu, self.std, self.y_dim, self.x_dim = self.params

    def compute_params(self):

        # Infer mu and std with fully connected layer
        model_mu = Layers(self.x)
        model_mu.conv2d(3, self.num_latent)
        mu = model_mu.get_output()
        model_var = Layers(self.x)
        model_var.conv2d(3, self.num_latent)
        std = tf.nn.softplus(model_var.get_output())

        # Get x dimensions
        h, w = self.x.get_shape()[1], self.x.get_shape()[2]

        # Make mu and std 4D for eq_samples and iw_samples
        std = tf.expand_dims(tf.expand_dims(std, 3), 3)
        mu = tf.expand_dims(tf.expand_dims(mu, 3), 3)
        return mu, std, h, w

    def compute_samples(self):
        """ Sample from a Normal distribution with inferred mu and std """
        eps = tf.random_normal([self.batch_size, self.eq_samples, self.iw_samples, self.x_dim, self.y_dim, self.num_latent])
        z = tf.reshape(eps * self.std + self.mu, [-1, self.x_dim, self.y_dim, self.num_latent])
        return z

    def log_likelihood(self, x, standard=False):
        """ Calculate Log Likelihood with particular mean and std
        x must be 2D. [batch_size * eqsamples* iwsamples, num_latent]
        """
        x_reshape = tf.reshape(x, [self.batch_size, self.eq_samples, self.iw_samples, self.x_dim, self.y_dim, self.num_latent])
        c = - 0.5 * math.log(2 * math.pi)
        if standard is False:
            density = c - tf.log(self.std + 1e-10) - (x_reshape - self.mu) ** 2 / (2 * self.std**2)
        else:
            density = c - x_reshape ** 2 / 2
        # sum over all importance weights. average over all eq_samples
        return tf.reduce_mean(tf.reduce_sum(density, axis=2), axis=(1, 2))


class BernoulliLayerFC(StochLayer):
    def __init__(self, x, num_latent, eq_samples, iw_samples=1):
        super().__init__(x, num_latent, eq_samples, iw_samples)
        self.mu = self.params

    def compute_params(self):
        # Infer mu (between 0 and 1) with fully connected layer
        model_mu = Layers(self.x)
        model_mu.fc(self.num_latent, activation_fn=None)  # To be applied later
        mu = model_mu.get_output()

        # Make mu and std 4D for eq_samples and iw_samples
        mu = tf.expand_dims(tf.expand_dims(mu, 1), 1)
        return mu

    def compute_samples(self):
        """ Sample from a Normal distribution with inferred mu and std """
        shape = [self.batch_size, self.eq_samples, self.iw_samples, self.num_latent]
        mu = tf.nn.sigmoid(self.mu)
        z = tf.select(tf.random_uniform(shape) - mu > 0, tf.ones(shape), tf.zeros(shape))
        return z

    def neg_log_likelihood(self, x):
        """ Calculate Log Likelihood with particular mean and std
        x must be 2D. [batch_size * eqsamples* iwsamples, num_latent]
        """
        x_reshape = tf.reshape(x, [self.batch_size, self.eq_samples, self.iw_samples, self.num_latent])
        density = tf.nn.sigmoid_cross_entropy_with_logits(self.mu, x_reshape)
        # sum over all importance weights. average over all eq_samples
        return tf.reduce_mean(tf.reduce_sum(-density, axis=2), axis=(1, 2))


class MultinomialLayerFC(StochLayer):
    def __init__(self, x, num_latent, eq_samples, iw_samples=1):
        super().__init__(x, num_latent, eq_samples, iw_samples)
        self.pi = self.params

    def compute_params(self):
        # Infer mu (between 0 and 1) with fully connected layer
        model_pi = Layers(self.x)
        model_pi.fc(self.num_latent, activation_fn=None)  # To be applied later
        pi = model_pi.get_output()

        # Make pi a 4D for eq_samples and iw_samples
        pi = tf.expand_dims(tf.expand_dims(pi, 1), 1)
        return pi

    def compute_samples(self):
        """ Sample from a Normal distribution with inferred mu and std """
        shape = [self.batch_size, self.eq_samples, self.iw_samples, self.num_latent]

        z = tf.nn.softmax(self.pi)
        return z

    def neg_log_likelihood(self, x, eps=1e-10):
        """ Calculate Log Likelihood with particular mean and std
        x must be 2D. [batch_size * eqsamples* iwsamples, num_latent]
        """
        x_reshape = tf.reshape(x, [self.batch_size, self.eq_samples, self.iw_samples, self.num_latent])

        # mean over the softmax outputs inside the log domain.
        pi = tf.reduce_mean(self.pi, axis=(1, 2))

        density = tf.reduce_mean(x * tf.log(pi + eps))
        return tf.reduce_sum(-density, axis=2)

