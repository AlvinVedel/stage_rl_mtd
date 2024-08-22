import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

class Dueling_QRDQN_model1(keras.Model):
    def __init__(self, input_dim1=220, input_dim2=6, n_actions=25, n_quantiles=51) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions = n_actions
        self.n_quantiles=n_quantiles
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.conv2 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.value = layers.Dense(1, activation='linear')
        self.avantages = layers.Dense(self.n_actions*self.n_quantiles, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        aug = tf.tile(tf.expand_dims(emb, axis=1), (1, self.input_dim1, 1))
        conc = layers.Concatenate(axis=2)([cone, aug])
        x = self.conv2(conc)
        x = layers.Flatten()(x)
        x = self.fc2(x)
        x = self.fc3(x)
        value = self.value(x)
        avantages = self.avantages(x)
        avantages_distr = layers.Reshape((self.n_actions, self.n_quantiles))(avantages)
        z_distrs = tf.expand_dims(value, axis=1) + avantages_distr - tf.reduce_mean(avantages_distr, axis=[1, 2], keepdims=True)
        return z_distrs




class Dueling_QRDQN_model2(keras.Model):
    def __init__(self, input_dim1=220, input_dim2=6, n_actions=25, n_quantiles=51) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions = n_actions
        self.n_quantiles = n_quantiles
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.value = layers.Dense(1, activation='linear')
        self.avantages = layers.Dense(self.n_actions*self.n_quantiles, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        flat = layers.Flatten()(cone)
        conc = layers.Concatenate(axis=1)([flat, emb])
        x = self.fc2(conc)
        x = self.fc3(x)
        value = self.value(x)
        avantages = self.avantages(x)
        avantages_distr = layers.Reshape((self.n_actions, self.n_quantiles))(avantages)
        z_distrs = tf.expand_dims(value, axis=1) + avantages_distr - tf.reduce_mean(avantages_distr, axis=[1, 2], keepdims=True)
        return z_distrs

    