from alvin_env import *
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, Model

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time 





seed = 42
gamma = 0.95#0.95  # Discount factor for past rewards
epsilon = 1.0  # Epsilon greedy parameter
epsilon_min = 0.1#0.2  # Minimum epsilon greedy parameter
epsilon_max = 1.0  # Maximum epsilon greedy parameter
epsilon_interval = (
    epsilon_max - epsilon_min
)  
print("la")


batch_size = 128  
max_steps_per_episode = 100

nb_agents = 1
num_actions = 9
grid_size_max=20
NWall=10

env = GridEnv(grid_size=grid_size_max, N=nb_agents, NWallMax=NWall)
env.reset()
def create_model(size=grid_size_max):
    input_commun = layers.Input(shape=(size, size), dtype=tf.int8, name="a")  
    input_passe = layers.Input(shape=(size, size, 1), name='passe')
    input_communX = layers.Input(shape=(4, ), name="b" ) 
    tbn = layers.Input(shape=( ), name="bn" ,dtype=tf.int32 )
    tartiflette = tbn[0]
    #batch_size = tf.shape(input_commun)[0] 
    #print(batch_size)

    encoding = tf.one_hot(tf.cast(input_commun, dtype=tf.int32), 3+3, name="c")
    encoding = tf.cast(encoding, dtype=tf.float32, name="d")
    indexes = tf.cast(tf.range(0, tartiflette), dtype=tf.int32, name="e")


    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(tartiflette)*3, dtype=tf.int32)), axis=1, name="f1")
    updates = tf.ones( (tartiflette), name="g")
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates, name="h")

    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(tartiflette)*4, dtype=tf.int32)), axis=1, name="f2")
    updates = input_communX[:, 2]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)

    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(tartiflette)*5, dtype=tf.int32)), axis=1, name="f3")
    updates = input_communX[:, 3]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)

   
    encoding = tf.concat((input_passe, encoding), axis=3)
    c1 = layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid", name="conv1")(encoding[..., 0:4])
    c2 = layers.Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu', padding="valid", name="conv2")(encoding[..., 4:])
    az = tf.concat((c1, c2 ), axis=3)


    conv_layer1 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid")(az)
    conv_layer2 = layers.Conv2D(96, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid")(conv_layer1)
    pool1 = layers.AveragePooling2D((2, 2))(conv_layer2)

    conv_layer3 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid")(pool1)
    conv_layer4 = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid")(conv_layer3)

    #pooling_layer2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer4)
    flatten_layer = layers.Flatten()(conv_layer4)
    layer1 = layers.Dense(256, activation='relu')(flatten_layer)
    layer2 = layers.Dense(256, activation='relu')(layer1)

   
    output = layers.Dense(9, activation='linear', name='global_output')(layer2)


    model = tf.keras.Model(inputs=[input_commun, input_communX, input_passe, tbn], outputs=output)
   
    print("model created")
    return model

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas
    
if __name__ == '__main__':

    NM = 1
    model_target_ = []
    
    model_target_.append(create_model())
    model_target_[-1].summary()

    if os.path.exists('ALVIN20_'+str(0)+'.keras') :
        modeltmp = tf.keras.models.load_model('ALVIN20_'+str(0)+'.keras')
        print("chargement modèle")
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

    average_rewards = []
    nb_steps = []
    print("run train")
    while True:  # Run until solved
        states = env.reset()

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
                     
                           action_probs = model_target_[i_%NM]([state_tensor1, state_tensor2, state_tensor3, tbn ], training=False)
                        
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
                 np.expand_dims(states_next[i_][0], axis=-1)), axis=2)
                d2=np.concatenate((states[i_][1], states_next[i_][1]), axis=0)
                d3=np.concatenate((states[i_][2], states_next[i_][2]), axis=2)

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

                    state2_samplev = sC[..., :4]#:2+nb_agents+3]
                    state2_next_samplev = sC[..., 4:]#2+nb_agents+3:]

                    state3_samplev = sD[..., 0]
                    states3_next_samplev = sD[..., 1]

                    aa=sB[:, 0]
                    rr=sB[:, 1]
                    dones=sB[:, 2]


                    t2 = time.time()
                    #future_reward = model_target([state_next_samplev, state_next_samplek], training=False)
                    future_reward = model_target_[team]([state_next_samplev,state2_next_samplev , states3_next_samplev, tbn], training=False)
                    
                    #future_reward    = model_target.predict([state_next_samplev, state_next_samplek], verbose=0)
                    updated_q_values = np.array(rr) +( gamma *  tf.reduce_max(future_reward, axis=1) )*(1-dones)

                    t3 = time.time()


                    masks = tf.one_hot(np.array(aa), num_actions)
                    with tf.GradientTape() as tape:
                        #q_values = model([np.array(state_samplev), np.array(state_samplek)] )
                        #q_values = model_target([np.array(state_samplev), state_samplek])#model
                        q_values = model_target_[team]([state_samplev, state2_samplev, state3_samplev, tbn])#model
                        
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
                print("DONE!", timestep, running_reward,  episode_count, frame_count, done, 'stats : ', env.iter, env.succeed)
                break
        if not done :
            print("PAS DONE!", timestep, running_reward,  episode_count, frame_count, done, ' stats : ', env.iter, env.succeed)

        avg_reward = np.mean(rewards)
        average_rewards.append(avg_reward)
        nb_steps.append(timestep)
        episode_count += 1
        if(episode_count%500==0):
            
            if episode_count <= 15000 :
                epochs = range(len(average_rewards))
                
                average_rewards_smoothed = moving_average(average_rewards, 50)
                epochs_smoothed = epochs[:len(average_rewards_smoothed)]
                plt.figure()
                plt.plot(epochs, average_rewards, color="grey", alpha=0.5)
                plt.plot(epochs_smoothed, average_rewards_smoothed, color='red')
                plt.xlabel('Époque')
                plt.ylabel('Récompense Moyenne')
                plt.ylim(-2, 15)
                plt.savefig('recompenses_20.png')
                plt.close() 

                nb_steps_smoothed = moving_average(nb_steps, 50)
                epochs_smoothed = epochs[:len(nb_steps_smoothed)]
                plt.figure()
                plt.plot(epochs, nb_steps, color='grey', alpha=0.5)
                plt.plot(epochs_smoothed, nb_steps_smoothed, color='red') 
                plt.xlabel('Époque')
                plt.ylabel('Nombre de steps')
                plt.ylim(0, 110)
                plt.savefig('steps_20.png')
                plt.close() 
            if(episode_count%80000==0 or episode_count%180000==0 or episode_count%360000==0):
                optimizer.learning_rate.assign(optimizer.learning_rate*0.8)

            for team in range(NM):
                model_target_[team].save('ALVIN20_'+str(team)+'.keras')
                model_target_[team].save_weights('WALVIN20_'+str(team)+'.weights.h5')

            print(" model save ",running_reward)

