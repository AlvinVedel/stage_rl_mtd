import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from variational_layer import VariationalLayer




class DQN_varia1_model1(keras.Model):
    """
    VERSION 1 DU DQN Variationnel ou la couche est juste après la fusion de modalité
    
    """
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.variational_dim=variational_dim
        self.n_actions = n_actions
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.conv2 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.q_val = layers.Dense(self.n_actions, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        aug = tf.tile(tf.expand_dims(emb, axis=1), (1, self.input_dim1, 1))
        conc = layers.Concatenate(axis=2)([cone, aug])
        x = self.conv2(conc)
        x = layers.Flatten()(x)
        x = self.varia(x)
        x = self.fc2(x)
        x = self.fc3(x)
        q_values = self.q_val(x)
        return q_values


class DQN_varia1_model2(keras.Model):
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.variational_dim=variational_dim
        self.n_actions = n_actions
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.q_val = layers.Dense(self.n_actions, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        flat = layers.Flatten()(cone)
        conc = layers.Concatenate(axis=1)([flat, emb])
        x = self.varia(conc)
        x = self.fc2(x)
        x = self.fc3(x)
        q_values = self.q_val(x)
        return q_values

    




class DQN_varia2_model1(keras.Model):
    """
    Version 2 du DQN variationnel dans lequel couche varia entre les 2 Dense de 512
    
    """
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.variational_dim=variational_dim
        self.n_actions = n_actions
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.conv2 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.fc3 = layers.Dense(512, activation='relu')
        self.q_val = layers.Dense(self.n_actions, activation='linear')

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
        x = self.varia(x)
        x = self.fc3(x)
        q_values = self.q_val(x)
        return q_values

class DQN_varia2_model2(keras.Model):
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions = n_actions
        self.variational_dim=variational_dim
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.fc3 = layers.Dense(512, activation='relu')
        self.q_val = layers.Dense(self.n_actions, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        flat = layers.Flatten()(cone)
        conc = layers.Concatenate(axis=1)([flat, emb])
        x = self.fc2(conc)
        x = self.varia(x)
        x = self.fc3(x)
        q_values = self.q_val(x)
        return q_values
    





class DQN_varia3_model1(keras.Model):
    """
    Version 3 du DQN variationnel ou la couche probabiliste est juste avant la prédiction des Q valeurs
    
    
    """
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions = n_actions
        self.variational_dim=variational_dim
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.conv2 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.q_val = layers.Dense(self.n_actions, activation='linear')

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
        x = self.varia(x)
        q_values = self.q_val(x)
        return q_values



class DQN_varia3_model2(keras.Model):
    def __init__(self, input_dim1=220, input_dim2=6, variational_dim=512, n_actions=25) :
        super().__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.n_actions = n_actions
        self.variational_dim=variational_dim
        self.conv1 = layers.Conv1D(filters=32, activation='relu', kernel_size=1, strides=1, padding='valid')
        self.fc1 = layers.Dense(32, activation='relu')
        self.fc2 = layers.Dense(512, activation='relu')
        self.fc3 = layers.Dense(512, activation='relu')
        self.varia = VariationalLayer(dim=self.variational_dim)
        self.q_val = layers.Dense(self.n_actions, activation='linear')

    def call(self, inputs) :
        input1, input2 = inputs
        oh = tf.one_hot(tf.cast(input1, dtype=tf.int32), 3)
        cone = self.conv1(oh)
        emb = self.fc1(input2)
        flat = layers.Flatten()(cone)
        conc = layers.Concatenate(axis=1)([flat, emb])
        x = self.fc2(conc)
        x = self.fc3(x)
        x = self.varia(x)
        q_values = self.q_val(x)
        return q_values