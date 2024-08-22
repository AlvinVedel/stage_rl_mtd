import os 
import tensorflow as tf
import numpy as np
from PIL import Image
from alvin_env import *
import gym
import os 
import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

nb_agents = 1
num_actions = 9
grid_size_max=10
NWall=10
env = GridEnv(grid_size=grid_size_max, N=nb_agents, NWallMax=NWall)

def resize_gif(input_path, output_path, new_size):
    # Ouvrir le GIF avec PIL
    gif = Image.open(input_path)

    # Redimensionner chaque frame du GIF
    resized_frames = []
    for frame in range(gif.n_frames):
        gif.seek(frame)
        resized_frame = gif.resize(new_size)
        resized_frames.append(resized_frame)

    # Sauvegarder le GIF redimensionné
    resized_frames[0].save(output_path, save_all=True, append_images=resized_frames[1:], loop=0, duration=100)


if __name__ == '__main__':

    NM = 1
    model_target_ = []
    
    model_target = tf.keras.models.load_model("./ALVIN20_0.keras")
    model_target.summary()


  
    #optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    max_memory_length = 100000
    max_memory_length_1 = max_memory_length
             


    memory_count_1 =[]
    superetatA_1 = []
    superetatB_1 = []
    superetatC_1=[]
    superetatD_1=[]
    memory_count_max_1 = []
    for i_ in range(NM):
        memory_count_1 +=[ 0]
        superetatA_1 +=[ np.zeros((max_memory_length,grid_size_max, grid_size_max, 2))]
        superetatC_1 +=[ np.zeros((max_memory_length,4*2))]#posS, posY, vitesse, orientation
        superetatD_1 +=[ np.zeros((max_memory_length, grid_size_max, grid_size_max, 2))]
        superetatB_1 +=[ np.zeros((max_memory_length,3))]#action, reward, done,
        memory_count_max_1 +=[0]



    episode_reward_history = []
    running_reward = 0
    episode_count = 0
    frame_count = 0         
    epsilon_random_frames = 1000#100 #5000
                            
    epsilon_greedy_frames = 150000#300000.0
                            
    update_after_actions = 4
    loss_function = keras.losses.Huber()
    max_steps_per_episode = 500


    print("run test")
    while True:  # Run until solved
        liste_frames = []
        states = env.reset(inference=True)
        base = np.zeros((grid_size_max, grid_size_max, 3), dtype=np.uint8)
        base[np.where(states[0][0] == 0)] = [255, 0, 0]
        base[np.where(states[0][0] == 1)] = [0, 255, 0]
        base[np.where(states[0][0] == 2)] = [255, 255, 255]
        visuel = base.copy()
        visuel[int(states[0][1][0])][int(states[0][1][1])] = [0, 0, 0]
        liste_frames.append(Image.fromarray(visuel))
        


        for timestep in range(1, max_steps_per_episode):
            frame_count += 1
            actions = []



            for i_ in range(nb_agents) :
                state_tensor1 = tf.convert_to_tensor(states[i_][0])
                state_tensor1 = tf.expand_dims(state_tensor1, 0)
                        
                state_tensor2 = tf.convert_to_tensor(states[i_][1])
                state_tensor2 = tf.expand_dims(state_tensor2, 0)

                state_tensor3 = tf.convert_to_tensor(states[i_][2])
                state_tensor3 = tf.expand_dims(state_tensor3, axis=0)

                tbn =  tf.convert_to_tensor(np.array([1]))
                        
                action_probs = model_target([state_tensor1, state_tensor2, state_tensor3, tbn ], training=False)
                            
                aa=tf.argmax(action_probs[0]).numpy()
                actions.append( aa)

                                
            states_next, reward, done, _ = env.step(actions)
            #liste_frames.append(states_next)
            running_reward=np.sum(reward) 

            visuel = base.copy()
            visuel[int(states_next[0][1][0])][int(states_next[0][1][1])] = [0, 0, 0]
            liste_frames.append(Image.fromarray(visuel))

            states = states_next
            if( done):
                #print("DONE!", timestep, running_reward,  episode_count, frame_count, done)
                break
        
        #print("PAS DONE!", timestep, running_reward,  episode_count, frame_count, done)
        #for image in liste_frames:
        #    print(type(image))
        input_path = "animations_inference/episode"+str(episode_count)+".gif"
        output_path = "animations_inference/episode"+str(episode_count)+"AUG.gif"
        liste_frames[0].save(input_path, format='GIF', append_images=liste_frames[1:], save_all=True, duration=250, loop=0)
        resize_gif(input_path, output_path, (100, 100))
  
        episode_count += 1
        print("episode terminé", timestep, running_reward,  episode_count, frame_count, done, "stats :", env.succeed, "/", env.iter)
