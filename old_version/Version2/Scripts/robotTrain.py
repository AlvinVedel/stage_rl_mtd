# %%
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import keras
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from PIL import Image
import math
from PIL import Image
from PIL import Image, ImageDraw
import random
import time
#import matplotlib.pyplot as plt
#import pandas as pd


from MonRobot import MonRobot
from Environnement import Environnement




# %%
nb_eps = 20000
nb_steps = 600

gamma = 0.95
loss_function = keras.losses.Huber()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)


# %%
size = 1
training = True
env = Environnement(size=size, classic=training)
print("env created")


if os.path.exists("./model"+str(40*size)+".h5"):
    print("récupération modèle")
    model = keras.models.load_model("./model"+str(40*size)+".h5")
    env.agent.main_network = model

#if os.path.exists("./dataframe.csv"):
#    print("récupération nb iter par ep")
#    dataframe = pd.read_csv("./dataframe.csv")



recording=False
##### OBJECTIF : 280, 30
#image_count = 0
print("START TRAINING")
for episode in range(nb_eps) :
    
    env.reset_env()

    if episode%10==0:
        episode_frames = []
        recording = True

    for step in range(nb_steps) :
       
        if recording :
            visuel_copy = np.array(env.visuel.copy())
            visuel_copy[int(env.agent.position[0])][int(env.agent.position[1])] = [0, 0, 0]
            episode_frames.append(Image.fromarray(visuel_copy))

        action = env.agent.choose_action(env.map, training)
        reward, done = env.step(action)
    
        if step % 4 == 0 and env.agent.nmemory > 64 and training :
            
            rdm_tran = env.agent.replay(32) 
           
            states = np.array(rdm_tran[0])
            next_states = np.array(rdm_tran[3])
            rewards = np.array(rdm_tran[2])
            actions = np.array(rdm_tran[1])
            dones = np.array(rdm_tran[4])
            previous_agent = np.array(rdm_tran[5])
            next_agent = np.array(rdm_tran[6])
            

            future_rewards = env.agent.main_network([next_states, next_agent], training=False)
            
            updated_q_values = rewards + gamma * tf.reduce_max(
                future_rewards, axis=2  # change to axis = 2 with 2 output
            )*(1 - dones)


            action_mask_angle = tf.one_hot([a[0] for a in actions], 3)
            action_mask_speed = tf.one_hot([a[1] for a in actions], 3)

        
            with tf.GradientTape() as tape:
                # Prédiction des Q-values pour les états actuels échantillonnés

                q_values_angle, q_values_speed = env.agent.main_network([states, previous_agent])   
                #q_values_angle = env.agent.main_network([states, previous_agent])  
               
 
                # Sélection des Q-values pour les actions prises
                q_action_angle = tf.reduce_sum(q_values_angle * action_mask_angle, axis=1)
                q_action_speed = tf.reduce_sum(q_values_speed * action_mask_speed, axis=1)

                # Calcul de la perte moyenne pour chaque sortie
                loss_angle = tf.reduce_mean(loss_function(updated_q_values, q_action_angle))
                loss_speed = tf.reduce_mean(loss_function(updated_q_values, q_action_speed))#a changer 
                # Perte totale
                #loss = loss_angle + loss_speed
                loss = loss_angle + loss_speed
 
           

            grads = tape.gradient(loss, env.agent.main_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, env.agent.main_network.trainable_variables))

        if done :
            break
    

    print(episode, " ",step, "  ", done , "  ",reward)
    if recording :
        if training :
            episode_frames[0].save("animations_half_continue/episode"+str(episode)+".gif", format='GIF', append_images=episode_frames[1:], save_all=True, duration=250, loop=0)
        else :
            episode_frames[0].save("animations_half_continue/episode"+str(episode)+"test.gif", format='GIF', append_images=episode_frames[1:], save_all=True, duration=250, loop=0)
        recording = False
        env.agent.main_network.save("model"+str(40*size)+".h5")

        episode_frames = []

    