import tensorflow as tf
from tensorflow.keras import layers


class Sampling(layers.Layer):
    """Uses (mu, logvar) to sample z, the vector encoding an observation."""
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config

class VariationalLayer(layers.Layer):
    def __init__(self, dim=512, **kwargs):
        super(VariationalLayer, self).__init__(**kwargs)
        self.dense_mu = layers.Dense(dim)
        self.dense_logvar = layers.Dense(dim)
        self.sampling = Sampling()

    def call(self, inputs):
        mu = self.dense_mu(inputs)
        logvar = self.dense_logvar(inputs)
        z = self.sampling((mu, logvar))
        self.add_loss(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)))
        return z

    def get_config(self):
        config = super(VariationalLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)