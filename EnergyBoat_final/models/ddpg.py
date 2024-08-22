import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers


class ActorModel(keras.Model) :
    def __init__(self, input_dim1=220, input_dim2=6, n_actions=2, amplitudes_actions=[1.5, 15]) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions=n_actions
        self.amplitudes = amplitudes_actions
        self.conv1 = layers.Conv1D(filters=32, kernel_size=1, activation='relu', strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.actions = layers.Dense(self.n_actions, activation='tanh')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        flat = layers.Flatten()(cone)
        emb = self.fc1(input2)
        x = layers.Concatenate(axis=1)([flat, emb])
        x = self.fc2(x)
        x = self.fc3(x)
        actions = self.actions(x)
        scaled_actions = actions * self.amplitudes
        return scaled_actions
    

class CriticModel(keras.Model):
    def __init__(self, input_dim1, input_dim2, n_actions):
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions=n_actions
        self.conv1 = layers.Conv1D(filters=32, kernel_size=1, activation='relu', strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(32, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.fc4 = layers.Dense(512, activation='relu')
        self.critic = layers.Dense(1, activation='linear')

    def call(self, inputs) :
        input1, input2, actions = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        flat = layers.Flatten()(cone)
        emb1 = self.fc1(input2)
        emb2 = self.fc2(actions)
        x = layers.Concatenate(axis=1)([flat, emb1, emb2])
        x = self.fc3(x)
        x = self.fc4(x)
        critic = self.critic(x)
        return critic
    

