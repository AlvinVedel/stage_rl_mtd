# %%
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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


from MonRobot2 import MonRobot
from Environnement2 import Environnement




# %%
nb_eps = 20000
nb_steps = 600

gamma = 0.95
loss_function = keras.losses.Huber()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

# %%
size = 2
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


# %%


#all_frames = []
#nb_ep = 289 
#i=0

#liste_graphe = []
#iter_par_ep = []
recording=False
##### OBJECTIF : 280, 30
#image_count = 0
print("START TRAINING")
for episode in range(nb_eps) :
    #nb_ep+=1
    #print("épisode : "+str(nb_ep))
    env.reset_env()

    #indices = np.where((np.array(env.visuel)[:,:,0] == 255) & (np.array(env.visuel)[:,:,1] == 255) & (np.array(env.visuel)[:,:,2] == 255))
    #print("objectif",indices)
    #indices = np.where(env.map[:,:] == 2)
    #print("objectif map",indices)
   
    #indices = np.where(env.map[:,:] == 2)


    #env.agent.replace_agent(point_depart)

    #nb_frames = 0
    #total_reward = 0
    if episode%10==0:
        episode_frames = []
        recording = True

    for step in range(nb_steps) :
        #print("agent en ", env.agent.position, "objectif en ", env.objectif, "distance :", env.previous_distance)
        if recording :
            visuel_copy = np.array(env.visuel.copy())
            #print("agent en ", env.agent.position, "objectif en ", env.objectif)
            visuel_copy[env.agent.position[0]][env.agent.position[1]] = [0, 0, 0]
            """visuel_copy[env.agent.position[0]+1][env.agent.position[1]] = [0, 0, 0]
            visuel_copy[env.agent.position[0]-1][env.agent.position[1]] = [0, 0, 0]
            visuel_copy[env.agent.position[0]][env.agent.position[1]+1] = [0, 0, 0]
            visuel_copy[env.agent.position[0]][env.agent.position[1]-1] = [0, 0, 0]"""
            episode_frames.append(Image.fromarray(visuel_copy))

        #start_time = time.time()
        #nb_frames+=1
        #print("step ", str(nb_frames), "distance restante : ", env.previous_distance)
        #print(env.agent.position)
        #previous_time = time.time()
        
        action = env.agent.choose_action(env.map, training)
        #print("temps action :", time.time() - previous_time)
        #previous_time = time.time()
        #here_time = time.time()
        #print("action choisie :", here_time-previous_time)
        # La greedy policy est définie dans la méthode choose action 
        # retour inutile
        reward, done = env.step(action)
        #print("temps step :", time.time() - previous_time)
        #previous_time = time.time()

        #here_time = time.time()
        #liste_graphe.append(reward)
        #print("step réalisé :", here_time-previous_time)
        # la fonction remember est déjà appelée dans step
        if step % 4 == 0 and env.agent.nmemory > 64 and training :
            #print("c'est une frame 4")
            #start_time = time.time()
            #previous_time = time.time()
            rdm_tran = env.agent.replay(32) 
            #print("temps replay :", time.time() - previous_time)
            #previous_time = time.time()
            #here_time = time.time()
            #print("replay memoire :", here_time-previous_time)
          
            #print(time.time() - start_time)
            #here_time = time.time()

           
            states = np.array(rdm_tran[0])
            next_states = np.array(rdm_tran[3])
            rewards = np.array(rdm_tran[2])
            actions = np.array(rdm_tran[1])
            dones = np.array(rdm_tran[4])
            previous_agent = np.array(rdm_tran[5])
            next_agent = np.array(rdm_tran[6])
            """
            for i in range(len(rdm_tran)):
                states.append(rdm_tran[i][0])
                next_states.append(rdm_tran[i][3])
                rewards.append(rdm_tran[i][2])
                actions.append(rdm_tran[i][1])
                dones.append(rdm_tran[i][4])
            """

            """
                states = np.array(i for rdm_tran[:,0])
                next_states = np.array(rdm_tran[:,3])
                rewards = np.array(rdm_tran[:,2])
                actions = np.array(rdm_tran[:,1])
                dones = np.array(rdm_tran[:,4])
                
            
            #print(rdm_tran.shape)
            #states_tf = tf.stack([tf.convert_to_tensor(state) for state in states])
            #next_states_tf = tf.stack([tf.convert_to_tensor(state) for state in next_states])
            #next_states_tf = tf.convert_to_tensor(next_states, dtype=object)
            #states_tf = tf.convert_to_tensor(states, dtype=object)
            next_states = np.array(next_states)
            dones = np.array(dones)
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            """
            #print(time.time() - here_time)
            #here_time = time.time()       
            #previous_time = time.time()    
            #print(next_states.shape)
            #print(next_agent.shape)
            #print(next_states.shape)
            #print(next_states)

            future_rewards = env.agent.main_network([next_states, next_agent], training=False)
            #here_time = time.time()
            #print("calcul futur reward :",  here_time-previous_time)
            
            #print(time.time() - here_time)
            #here_time = time.time() 
            #previous_time = time.time()
            # Q value = reward + discount factor * expected future reward : comme dans DQN keras
            updated_q_values = rewards + gamma * tf.reduce_max(
                future_rewards, axis=2  # change to axis = 2 with 2 output
            )*(1 - dones)

            #here_time = time.time()
            #print("q value updated :",  here_time-previous_time)  
            #print(time.time() - here_time)
            #here_time = time.time() 
            # 2 masques pour chacune de mes variables


            action_mask_angle = tf.one_hot([a[0] for a in actions], 4)
            action_mask_speed = tf.one_hot([a[1] for a in actions], 4)
            #action_mask_angle = tf.one_hot([a for a in actions], 4)





            #print(time.time() - here_time)
            #here_time = time.time() 
            #previous_time = time.time()
            with tf.GradientTape() as tape:
                # Prédiction des Q-values pour les états actuels échantillonnés

                q_values_angle, q_values_speed = env.agent.main_network([states, previous_agent])   # LIGNE EN ERREUR
                #q_values_angle = env.agent.main_network([states, previous_agent])   # LIGNE EN ERREUR
               
 
                # Sélection des Q-values pour les actions prises
                q_action_angle = tf.reduce_sum(q_values_angle * action_mask_angle, axis=1)
                q_action_speed = tf.reduce_sum(q_values_speed * action_mask_speed, axis=1)

                # Calcul de la perte moyenne pour chaque sortie
                loss_angle = tf.reduce_mean(loss_function(updated_q_values, q_action_angle))
                loss_speed = tf.reduce_mean(loss_function(updated_q_values, q_action_speed))#a changer 
                # Perte totale
                #loss = loss_angle + loss_speed
                loss = loss_angle + loss_speed

            #here_time = time.time()
            #print("calcul loss :",  here_time-previous_time)
            #print(time.time() - here_time)
            #here_time = time.time() 
            # Calcul du gradient et application des mises à jour des poids
            #previous_time = time.time()

            grads = tape.gradient(loss, env.agent.main_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, env.agent.main_network.trainable_variables))

            #here_time = time.time()
            #print("gradient appliqué :",  here_time-previous_time)
            #print(time.time() - here_time)
            #here_time = time.time() 
            #print("temps propa :", time.time() - previous_time)
            #previous_time = time.time()

        #if nb_frames % 5000 == 0 :
        #    #print("maj model")
        #    env.agent.update_target_model()

        if done :
            break
        
        #episode_frames.append(env.visuel)
        #print("gif output")
        #name = "./gifs_deplacements/image test"+str(nb_frames)+" : "+str(env.agent.orientation)+" "+str(env.agent.vitesse)+'.png'
        #env.visuel.save(name)
        #episode_frames[0].save(name, save_all=True, append_images=episode_frames[1:], duration=1500, loop=0)
        #print("enf of step ", str(nb_frames))
        #if nb_frames % 2500 == 0 :
        #    print("gif output")
        #    name = "./gifs_deplacements/animation"+str(nb_ep)+" 2500x"+str(i)+'.gif'
        #    episode_frames[0].save(name, save_all=True, append_images=episode_frames[1:], duration=150, loop=0)
        #    i+=1
        #    episode_frames = []
    print(episode, " ",step, "  ", done , "  ",reward)
    if recording :
        if training :
            episode_frames[0].save("animations_3/episode"+str(episode)+".gif", format='GIF', append_images=episode_frames[1:], save_all=True, duration=250, loop=0)
        else :
            episode_frames[0].save("animations_3/episode"+str(episode)+"test.gif", format='GIF', append_images=episode_frames[1:], save_all=True, duration=250, loop=0)
        recording = False
        env.agent.main_network.save("model"+str(40*size)+".h5")

        episode_frames = []

    """
    plt.figure()
    plt.plot(liste_graphe)
    #iter_par_ep.append(len(liste_graphe))
    plt.xlabel('Itérations')
    plt.ylabel('Reward')
    plt.title('Évolution de la valeur au fil des itérations')
    plt.savefig('./graphiques/evolution_reward'+str(nb_ep)+'.png')  # Nom du fichier et format
    liste_graphe = []
    df = pd.DataFrame([len(liste_graphe)], columns=['Itérations'])
    if 'dataframe' in globals() :
        dataframe = pd.concat([dataframe, df], ignore_index=True)
    else:
        dataframe = df.copy()
    dataframe.to_csv("dataframe.csv")


    if len(episode_frames)> 2 :
        name = "./gifs_deplacements/animation"+str(nb_ep)+" fin"+str(i)+'.gif'
        episode_frames[0].save(name, save_all=True, append_images=episode_frames[1:], duration=150, loop=0)
    #env.agent.update_target_model()
    env.agent.main_network.save("sauvegarde_model.h5")
    """
    """if episode_frames:
        if (nb_ep % 10 ) + 1 == 1 :        
            name = "./gifs_deplacements/animation"+str(nb_ep), str(i)+'_fin.gif'
            episode_frames[0].save(name, save_all=True, append_images=episode_frames[1:], duration=150, loop=0)
    else:
        print("No frames captured, unable to save animation.")
    all_frames.append(episode_frames)
    """
        

"""

plt.plot(iter_par_ep)
plt.xlabel('Episode')
plt.ylabel('Nb itérations')
plt.title("Évolution du nombre d'itération par épisode au fil des itérations")
plt.savefig('./graphiques/nombre_iterations.png')  # Nom du fichier et format
"""

"""
big_gif = []
for episode in all_frames :
    for frame in episode :
        big_gif.append(frame)

big_gif[0].save("./complete_training.gif", save_all=True, append_images=big_gif[1:], duration=200, loop=0)
"""

# %%
        
#episode_frames[0].save("./animation.gif", save_all=True, append_images=episode_frames[1:], duration=10*len(episode_frames), loop=0)



#experimentation conditions array
"""
arr = np.array([[[2, 3], [3, 6], [4, 1]], [[2, 1], [3, 1], [4, 1]], [[2, 7], [3, 6], [4, 0]]])

#coords = np.where(arr[:, :, 1]==7)
#print(coords[0][0], coords[1][0])

image_arr = np.array(image)
robot_point = (190, 390)

indices_obj = np.argwhere((image_arr[:, :, 1] == 255) & (image_arr[:, :, 0]==255) & (image_arr[:, :, 2]==255))
distances = np.linalg.norm(indices_obj - robot_point, axis=1)
moyenne_distance = np.mean(distances)

print(image_arr[0, 1][2])

print(moyenne_distance)
"""



# %%
