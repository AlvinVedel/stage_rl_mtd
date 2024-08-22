import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from VAE import CVAE
from tensorflow.keras.models import load_model
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
Adaptation de la documentation Tensorflow sur le Convolutional Variational Auto Encoder (CVAE)
"""


try :
    base = np.load("./cones_samples.npy")
except Exception as e:
    print("Pas de samples d'entrainements trouvés")
    sys.exit(1)


indices = np.arange(base.shape[0])
np.random.shuffle(indices)

train_indices = indices[:int(base.shape[0] * 0.9)]
test_indices = indices[int(base.shape[0] * 0.9):]
train_base = base[train_indices]
test_base = base[test_indices]
train_base = tf.one_hot(train_base, 3)
test_base = tf.one_hot(test_base, 3)

batch_size=32
optimizer = tf.keras.optimizers.Adam(1e-4)


train_dataset = (tf.data.Dataset.from_tensor_slices(train_base)
                 .shuffle(440000).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_base)
                .shuffle(40000).batch(batch_size))


epochs=50
latent_dim=256


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    reconstruction_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    rc_loss1 = reconstruction_loss_fn(x, x_logit)
    mse_loss = tf.keras.losses.MeanSquaredError()
    rc_loss2 = mse_loss(x, tf.sigmoid(x_logit))
    return -tf.reduce_mean(logpx_z + logpz - logqz_x), rc_loss1, rc_loss2

def compute_reconstruction_loss(model, x):
      mean, logvar = model.encode(x)
      z = model.reparameterize(mean, logvar)
      x_logit = model.decode(z)
      reconstruction_loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
      reconstruction_loss = reconstruction_loss_fn(x, x_logit)
      mse_loss = tf.keras.losses.MeanSquaredError()
      rc_loss2 = mse_loss(x, tf.sigmoid(x_logit))
      return reconstruction_loss, rc_loss2

@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
      loss, rc_l1, rc_l2  = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, rc_l1, rc_l2





model = CVAE(latent_dim)
history = {"train_loss": np.zeros(epochs), "train_cce" : np.zeros(epochs), "train_mse":np.zeros(epochs),
    "test_loss": np.zeros(epochs), "test_cce" : np.zeros(epochs), "test_mse":np.zeros(epochs)}


for epoch in range(1, epochs + 1):
    start_time = time.time()
    loss_values = []
    cce_values = []
    mse_values = []
    for train_x in train_dataset:
      loss, rc_l1, rc_l2 = train_step(model, train_x, optimizer)
      loss_values.append(-loss)
      cce_values.append(rc_l1)
      mse_values.append(rc_l2)
    end_time = time.time()
    history["train_loss"][epoch-1] = np.mean(loss_values)
    history["train_cce"][epoch-1] = np.mean(cce_values)
    history["train_mse"][epoch-1] = np.mean(mse_values)

    
    test_loss = []
    test_cce = []
    test_mse = []
    for test_x in test_dataset:
        loss, rc_l1, rc_l2 = compute_loss(model, test_x)
        test_loss.append(-loss)
        test_cce.append(rc_l1)
        test_mse.append(rc_l2)  
    elbo = np.mean(test_loss)
    history["test_loss"][epoch-1] = elbo
    history["test_cce"][epoch-1] = np.mean(test_cce)
    history["test_mse"][epoch-1] = np.mean(test_mse)

    print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))


x = list(range(epochs))
plt.plot(x, history["train_loss"], marker='o', linestyle='-', color='b', label="train loss")
plt.plot(x, history["test_loss"], marker='o', linestyle='-', color='r', label='test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('ELBO VAE train/test')
plt.grid(True)
plt.savefig("loss_VAE_"+str(latent_dim)+".png")
plt.close()

plt.plot(x, history["train_cce"], marker='o', linestyle='-', color='b', label='train cce')
plt.plot(x, history["test_cce"], marker='o', linestyle='-', color='r', label='test cce')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('CCE VAE train/test')
plt.grid(True)
plt.savefig("cce_VAE_"+str(latent_dim)+".png")
plt.close()

plt.plot(x, history["train_mse"], marker='o', linestyle='-', color='b', label="train mse")
plt.plot(x, history["test_mse"], marker='o', linestyle='-', color='r', label='test mse')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MSE VAE train/test')
plt.grid(True)
plt.savefig("mse_VAE_"+str(latent_dim)+".png")
plt.close()


model.encoder.save("encoder_VAE_"+str(latent_dim)+".h5")
model.decoder.save("decoder_VAE_"+str(latent_dim)+".h5")

np.save("test_base.npy",test_base)



