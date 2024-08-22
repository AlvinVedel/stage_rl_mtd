import numpy as np
import tensorflow as tf 
from tensorflow.keras import layers

def build_encoder(latent_dim) :
    input_commun = layers.Input(shape=(220,3), dtype=tf.float32, name="a")
    #encoding = tf.one_hot(tf.cast(input_commun, dtype=tf.int32), 3, name="c")   # batch, 220, 3
    c1 = layers.Conv1D(32, kernel_size=1, strides=1, activation='relu', padding='valid', name='c1')(input_commun) # batch, 220, 32

    #p1 = layers.AveragePooling1D(pool_size=2, padding='valid')(c2) # batch, 110, 32
    flat = layers.Flatten()(c1)  # b, 7040
    d1 = layers.Dense(1024, activation='relu', name='d1')(flat)
    d2 = layers.Dense(2*latent_dim, name='d2')(d1) # b, 128

    model = tf.keras.Model(inputs=input_commun, outputs=d2)
    return model

def build_decoder(latent_dim):
  input = layers.Input(shape=(latent_dim), dtype=tf.float32)
  d1 = layers.Dense(3072, activation='relu', name='d3')(input)
  d2 = layers.Dense(660, name='d4')(d1)
  d2 = layers.Reshape(target_shape=(220, 3))(d2)  # b, 110, 32
  model = tf.keras.Model(inputs=input, outputs=d2)
  return model


class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim=128):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = build_encoder(latent_dim)
    self.decoder = build_decoder(latent_dim)

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return tf.sigmoid(self.decode(eps))

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z):
    logits = self.decoder(z)
    return logits
  

