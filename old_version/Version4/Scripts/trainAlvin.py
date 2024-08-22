from alvin_env import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, Model

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time 





seed = 42
gamma = 0.95#0.95  # Discount factor for past rewards
epsilon = 1  # Epsilon greedy parameter
epsilon_min = 0.1#0.2  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  
print("la")


batch_size = 128  
max_steps_per_episode = 240

nb_agents = 1
num_actions = 9
grid_size_max=80
NWall=10

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas

env = GridEnv(grid_size=grid_size_max, N=nb_agents, NWallMax=NWall)
env.reset()
def create_model(size=819):
    input_commun = layers.Input(shape=(size), dtype=tf.int8, name="a")  
    input_passe = layers.Input(shape=(size, 1), dtype=tf.float32, name='passe')
    input_communX = layers.Input(shape=(3, ), name="b" )   # Infos supp : angle objectif, angle robot, vitesse robot
    #tbn = layers.Input(shape=( ), name="bn" ,dtype=tf.int32 )
    #tartiflette = tbn[0]
    #batch_size = tf.shape(input_commun)[0] 
    #print(batch_size)

    encoding = tf.one_hot(tf.cast(input_commun, dtype=tf.int32), 5, name="c")   # On onehot à 0 mur 1 chemin  2 objectif (plus de robot, vitesse / angle passé à coté) 3 pour deja passe 4 pour obstacles
    encoding = tf.concat((input_passe, encoding), axis=2)

    c1 = layers.Conv1D(32, kernel_size=1, strides=1, activation='relu', padding='valid', name='c1')(encoding)
    l1 = layers.Dense(32, activation='relu', name='l1')(input_communX)
    l1 = tf.expand_dims(l1, axis=1)

    l1 = tf.tile(l1, multiples=[1, size, 1])
    b = tf.concat((c1, l1), axis=-1)


    c2 = layers.Conv1D(32, kernel_size=1, strides=1, activation='relu', padding='valid', name='c2')(b)

    flatten_layer = layers.Flatten()(c2)
    

    layer1 = layers.Dense(1024, activation='relu')(flatten_layer)
    layer2 = layers.Dense(1024, activation='relu')(layer1)



    output = layers.Dense(9, activation='linear', name='global_output')(layer2)

    model = tf.keras.Model(inputs=[input_commun, input_communX, input_passe], outputs=output)
   
    print("model created")
    return model


    
if __name__ == '__main__':

    NM = 1
    model_target_ = []
    
    model_target_.append(create_model())
    model_target_[-1].summary()
    if os.path.exists('ALVIN3a_'+str(0)+'.keras'):
        modeltmp = tf.keras.models.load_model('ALVIN3a_'+str(0)+'.keras')
        layer_names = [layer.name for layer in modeltmp.layers]
        l=layer_names 
        for i in l:
            model_target_[-1].get_layer(i).set_weights(modeltmp.get_layer(i).get_weights())
            print("chargement ", i)
    
  
    optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    max_memory_length = 100000
    max_memory_length_1 = max_memory_length
             


    memory_count_1 =[]
    superetatA_1 = []
    superetatB_1 = []
    superetatC_1=[]
    superetatD_1=[]
    memory_count_max_1 = []

    poids = np.ones((max_memory_length))


    for i_ in range(NM):
        memory_count_1 +=[ 0]
        superetatA_1 +=[ np.zeros((max_memory_length, 819, 2))]
        superetatC_1 +=[ np.zeros((max_memory_length,3*2))]#vitesse, orientation, direction obj
        superetatD_1 +=[ np.zeros((max_memory_length, 819, 2))]
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
    seed = np.random.randint(0, 2**32)    

    print("run train")
    average_rewards = []
    nb_steps = []
    while True:  # Run until solved
        states = env.reset(obstacles=2)

        isalways_best_choice = np.random.rand(1)[0]<0.05
        checkispossible = np.random.rand(1)[0]<0.25
        randomIsX2 = np.random.rand(1)[0]<0.25

        for timestep in range(1, max_steps_per_episode):
            frame_count += 1
            actions = []
            rewards = []

            choix_random = False
            choix_randomvrai = False
            choix_best = False
            choix_proba = False


            tA = time.time()

            for i_ in range(nb_agents):
                  if(isalways_best_choice):
                     choix_best = True
                  elif frame_count < epsilon_random_frames or epsilon+randomIsX2*epsilon > np.random.rand(1)[0]:
                     if((np.random.random()<0.9 and frame_count < epsilon_random_frames//2) or np.random.random()<0.9 ):
                           choix_randomvrai = True
                     else:
                           choix_random = True
                  else: 
                     choix_proba=True


                  if(choix_randomvrai ):#or i_%2==0):
                            actions.append(np.random.choice(num_actions))
                  elif(choix_random):
                            actions.append(np.random.choice(num_actions))
                  else:
                           state_tensor1 = tf.convert_to_tensor(states[i_][0])
                           state_tensor1 = tf.expand_dims(state_tensor1, 0)
                           
                           state_tensor2 = tf.convert_to_tensor(states[i_][1])
                           state_tensor2 = tf.expand_dims(state_tensor2, 0)

                           state_tensor3 = tf.convert_to_tensor(states[i_][2])
                           state_tensor3 = tf.expand_dims(state_tensor3, 0)

                           tbn =  tf.convert_to_tensor(np.array([1]))
                     
                           action_probs = model_target_[i_%NM]([state_tensor1, state_tensor2, state_tensor3 ], training=False)
                        
                           aa=tf.argmax(action_probs[0]).numpy()
                           actions.append( aa)

                          

                    
            states_next, reward, done, _ = env.step(actions)
            rewards.append(reward)
            
            
            tB = time.time()


            epsilon -= epsilon_interval / epsilon_greedy_frames
            epsilon = max(epsilon, epsilon_min)
            
            # Apply the sampled action in our environment
            #print("-->", timestep,  reward, done, aa )
            running_reward=np.sum(reward)
            tC = time.time()

            # Save actions and states in replay buffer
            for i_ in range(nb_agents):
                d=np.concatenate((np.expand_dims(states[i_][0], axis=-1),
                 np.expand_dims(states_next[i_][0], axis=-1)), axis=1)
                d2=np.concatenate((states[i_][1], states_next[i_][1]), axis=0)
                d3=np.concatenate((np.expand_dims(states[i_][2], axis=-1),
                 np.expand_dims(states_next[i_][2], axis=-1)), axis=1)

                poids[memory_count_1[i_%NM]]=1
                if(done):
                    for j__ in range(0, timestep):
                        j_ = memory_count_1[i_%NM]-j__
                        if(j_<0):
                            j_+=max_memory_length_1
                        poids[j_]=10 + max(0, 10 - j__/2.0)

                
                superetatA_1[i_%NM][memory_count_1[i_%NM]] = d
                superetatC_1[i_%NM][memory_count_1[i_%NM]] = d2
                superetatD_1[i_%NM][memory_count_1[i_%NM]] = d3
                superetatB_1[i_%NM][memory_count_1[i_%NM]] = np.array([actions[i_], reward, done])
                memory_count_1[i_%NM]=(memory_count_1[i_%NM]+1)%max_memory_length_1
                memory_count_max_1[i_%NM]=min(memory_count_max_1[i_%NM]+1,max_memory_length_1-1 )

            states = states_next
            tD = time.time()

            if frame_count % update_after_actions == 0 \
                and memory_count_max_1[0] > batch_size*3  :
                tbn =  tf.convert_to_tensor(np.array([batch_size]))

                for team in range(0, NM):
                    t1 = time.time()
                    indices            = np.random.choice(range(memory_count_max_1[team]), size=batch_size, p=poids[:memory_count_max_1[team]]/np.sum(poids[:memory_count_max_1[team]]))
                    sB = superetatB_1[team][indices]
                    sA = superetatA_1[team][indices]
                    sC = superetatC_1[team][indices]
                    sD = superetatD_1[team][indices]
                    state_samplev = sA[..., 0]
                    state_next_samplev = sA[..., 1]

                    state2_samplev = sC[..., :3]#:2+nb_agents+3]
                    state2_next_samplev = sC[..., 3:]#2+nb_agents+3:]

                    state3_samplev = sD[..., 0]
                    states3_next_samplev = sD[..., 1]

                    aa=sB[:, 0]
                    rr=sB[:, 1]
                    dones=sB[:, 2]


                    t2 = time.time()
                    #future_reward = model_target([state_next_samplev, state_next_samplek], training=False)
                    #print(state_next_samplev.shape, state2_next_samplev.shape, states3_next_samplev.shape)
                    future_reward = model_target_[team]([state_next_samplev,state2_next_samplev , states3_next_samplev], training=False)
                    
                    #future_reward    = model_target.predict([state_next_samplev, state_next_samplek], verbose=0)
                    updated_q_values = np.array(rr) +( gamma *  tf.reduce_max(future_reward, axis=1) )*(1-dones)

                    t3 = time.time()


                    masks = tf.one_hot(np.array(aa), num_actions)
                    with tf.GradientTape() as tape:
                        #q_values = model([np.array(state_samplev), np.array(state_samplek)] )
                        #q_values = model_target([np.array(state_samplev), state_samplek])#model
                        q_values = model_target_[team]([state_samplev, state2_samplev, state3_samplev])#model
                        
                        q_action = tf.reduce_sum(tf.multiply(q_values,masks), axis=1)
                        #print(q_action, updated_q_values)

                        loss = loss_function(updated_q_values, q_action)
                        #print(loss.shape,q_action.shape, state_samplek.shape )
                        #exit()
                    t4 = time.time()
                    grads = tape.gradient(loss, model_target_[team].trainable_variables)#model
                    optimizer.apply_gradients(zip(grads, model_target_[team].trainable_variables))#model
                    
                    tE = time.time()
                    #print("A", tE-tD, tD-tC, tC-tB, tB-tA)
                    #print("B", tE-t4, t4-t3, t3-t2, t2-t1)

                
            if( done):
                if reward==-10 :
                    print("MORT!", timestep, running_reward,  episode_count, frame_count, done, 'stats : ', env.iter, env.succeed)
                    break
                else:
                    print("DONE!", timestep, running_reward,  episode_count, frame_count, done, 'stats : ', env.iter, env.succeed)
                    break
            
        if not done :
            print("PAS DONE!", timestep, running_reward,  episode_count, frame_count, done, ' stats : ', env.iter, env.succeed)
        else :
            seed = np.random.randint(0, 2**32)    

        avg_reward = np.mean(rewards)
        average_rewards.append(avg_reward)
        nb_steps.append(timestep)

        episode_count += 1
        if(episode_count%500==0):
            if episode_count <= 50000 :
                epochs = range(len(average_rewards))
                
                average_rewards_smoothed = moving_average(average_rewards, 50)
                epochs_smoothed = epochs[:len(average_rewards_smoothed)]
                plt.figure()
                plt.plot(epochs, average_rewards, color="grey", alpha=0.5)
                plt.plot(epochs_smoothed, average_rewards_smoothed, color='red')
                plt.xlabel('Époque')
                plt.ylabel('Récompense Moyenne')
                plt.ylim(-12, 12)
                plt.savefig('recompenses_obstacles40.png')
                plt.close() 

                nb_steps_smoothed = moving_average(nb_steps, 10)
                epochs_smoothed = epochs[:len(nb_steps_smoothed)]
                plt.figure()
                plt.plot(epochs, nb_steps, color='grey', alpha=0.5)
                plt.plot(epochs_smoothed, nb_steps_smoothed, color='red') 
                plt.xlabel('Époque')
                plt.ylabel('Nombre de steps')
                plt.ylim(0, 250)
                plt.savefig('steps_obstacles40.png')
                plt.close() 
            if(episode_count%80000==0 or episode_count%180000==0 or episode_count%360000==0):
                optimizer.learning_rate.assign(optimizer.learning_rate*0.8)

            for team in range(NM):
                model_target_[team].save('ALVIN3a_'+str(team)+'.keras')
                model_target_[team].save_weights('WALVIN3a_'+str(team)+'.weights.h5')

            print(" model save ",running_reward)

