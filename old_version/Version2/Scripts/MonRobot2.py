import tensorflow as tf
import keras.layers as layers
import numpy as np
import math
import random
from PIL import Image



def create_model(size):
    input_commun = layers.Input(shape=(40*size, 40*size), dtype=tf.int8, name="a")  
    input_communX = layers.Input(shape=(4, ), name="b" ) 
    batch_size = tf.shape(input_commun)[0]
    encoding = tf.one_hot(tf.cast(input_commun, dtype=tf.int32), 3+3, name="c")
    encoding = tf.cast(encoding, dtype=tf.float32, name="d")
    indexes = tf.cast(tf.range(0, batch_size), dtype=tf.int32, name="e")


    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(batch_size)*3, dtype=tf.int32)), axis=1, name="f1")
    updates = tf.ones( (batch_size,), name="g")
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates, name="h")


    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(batch_size)*4, dtype=tf.int32)), axis=1, name="f2")
    updates = input_communX[:, 2]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)
    indices = tf.stack((indexes, 
                        tf.cast(input_communX[:,  0], dtype=tf.int32), 
                        tf.cast(input_communX[: , 1], dtype=tf.int32), 
                        tf.cast(tf.ones(batch_size)*5, dtype=tf.int32)), axis=1, name="f3")
    updates = input_communX[:, 3]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)
    
    #max_stride = np.ceil(size/2)
    #if max_stride == 1 :
    #    max_stride = 3
    """
    test = tf.make_ndarray(tf.constant(encoding))
    img0 = test[:, :, 0]
    img_array = np.array(img0)
    tf.keras.preprocessing.image.save_img('pixel0.png', img_array)
    img1 = test[:, :, 1]
    img_array =np.array(img1)
    tf.keras.preprocessing.image.save_img('pixel1.png', img_array)
    img2 = test[:, :, 2]
    img_array = np.array(img2)
    tf.keras.preprocessing.image.save_img('pixel2.png', img_array)
    img3 = test[:, :, 3]
    img_array = np.array(img3)
    tf.keras.preprocessing.image.save_img('pixel3.png', img_array)
    img4 = test[:, :, 4]
    img_array = np.array(img4)
    tf.keras.preprocessing.image.save_img('pixel4.png', img_array)
    img5 = test[:, :, 5]
    img_array = np.array(img5)
    tf.keras.preprocessing.image.save_img('pixel5.png', img_array)
    """


    conv_layer1 = layers.Conv2D(64*4, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding="valid")(encoding)

    for i in range(size):
        conv_layer1 = layers.Conv2D(32*4, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv_layer1)
    
    
    conv_layer4 = layers.Conv2D(32*4, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv_layer1)
    
    pooling_layer2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer4)
    flatten_layer = layers.Flatten()(pooling_layer2)
    layer = layers.Dense(1024, activation='relu')(flatten_layer)
   


   
    output = layers.Dense(9, activation='linear', name='global_output')(layer)


    model = tf.keras.Model(inputs=[input_commun, input_communX], outputs=output)
    model2 = tf.keras.Model(inputs=[input_commun, input_communX], outputs=encoding)
   
    print("model created")
    return model, model2
"""
def create_model(batch_size=32):
    print("creating model...")
    input_commun = layers.Input(shape=(400, 400), dtype=tf.int8)  # Entrée pour les images
    input_communX = layers.Input(shape=(4, ) )  # Entrée pour les images        x, y, v, o = input_communX
    #print(input_commun.shape)
    ######                       disons 0 pour rouge, 1 pour vert , 2 pour objectif
    encoding = tf.one_hot(tf.cast(input_commun, dtype=tf.int32), 4+2)
    encoding = tf.cast(encoding, dtype=tf.float32)

    #batch_size_variable = tf.Variable(32, dtype=tf.int32, trainable=False, validate_shape=False)  # initialisation avec une taille de lot de 32
    #batch_size = batch_size_variable.numpy()
    
    indices = tf.stack((tf.cast(tf.range(0, batch_size), dtype=tf.int32), tf.cast(input_communX[:, 0], dtype=tf.int32), tf.cast(input_communX[: , 1], dtype=tf.int32), tf.cast(tf.ones(batch_size)*4, dtype=tf.int32)), axis=1)(batch_size)
    updates = tf.constant(1.0, shape=(batch_size,))
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)
    
    indices = tf.stack((tf.cast(tf.range(0, batch_size), dtype=tf.int32), tf.cast(input_communX[:, 0], dtype=tf.int32), tf.cast(input_communX[: , 1], dtype=tf.int32), tf.cast(tf.ones(batch_size)*5, dtype=tf.int32)), axis=1)
    updates = input_communX[:, 2]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)

    indices = tf.stack((tf.cast(tf.range(0, batch_size), dtype=tf.int32), tf.cast(input_communX[:, 0], dtype=tf.int32), tf.cast(input_communX[: , 1], dtype=tf.int32), tf.cast(tf.ones(batch_size)*6, dtype=tf.int32)), axis=1)
    updates = input_communX[:, 3]
    encoding = tf.tensor_scatter_nd_add(encoding, indices, updates)
    
    
    conv_layer1 = layers.Conv2D(64*4, kernel_size=(7, 7), strides=(3, 3), activation='relu')(encoding)
    conv_layer2 = layers.Conv2D(64*4, kernel_size=(6, 6), strides=(3, 3), activation='relu')(conv_layer1)
    
    conv_layer3 = layers.Conv2D(32*4, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv_layer2)
    conv_layer4 = layers.Conv2D(32*4, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv_layer3)
    pooling_layer2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer4)
    flatten_layer = layers.Flatten()(pooling_layer2)
    layer = layers.Dense(64*4, activation='relu')(flatten_layer)
    layer2a = layers.Dense(64*4, activation='relu')(layer)
    layer2b = layers.Dense(64*4, activation='relu')(layer)


    angle_output = layers.Dense(4, activation='linear', name='angle_output')(layer2a)
    speed_output = layers.Dense(4, activation='linear', name='speed_output')(layer2b)

    model = tf.keras.Model(inputs=[input_commun, input_communX], outputs=[angle_output, speed_output])
    #model.compile(optimizer='Adadelta', loss={'angle_output': 'binary', 'speed_output': 'mean_squared_error'})
    print("model created")
    return model
"""
"""
def rotate_point(point, angle_degrees, centre):
        angle_radians = math.radians(angle_degrees)    
        translated_point = (point[0] - centre[0], point[1] - centre[1])
        rotated_x = translated_point[0] * math.cos(angle_radians) - translated_point[1] * math.sin(angle_radians)
        rotated_y = translated_point[0] * math.sin(angle_radians) + translated_point[1] * math.cos(angle_radians)
        
        # Déplacer le point de retour à sa position d'origine
        rotated_point = (rotated_x + centre[0], rotated_y + centre[1])
        return rotated_point
"""






class MonRobot:
    def __init__(self, position, orientation=0, vitesse=2, size=10):
        self.position = position  
        self.orientation = orientation 
        self.vitesse = vitesse 
        self.height = 20 
        self.width=10
       
        self.main_network, self.encoder = create_model(size)
        
        self.memory_max_size = 10000
        self.step_count = 0
        self.epsilon = 1
        self.min_epsilon = 0.07
        self.epsilon_factor = 0.95
 
        self.dimMap = 40*size

        self.sampled_previous_states = np.zeros((self.memory_max_size, self.dimMap, self.dimMap))
        self.sampled_actions = np.zeros((self.memory_max_size), dtype=np.int8)
        self.rewards = np.zeros((self.memory_max_size))
        self.next_states = np.zeros((self.memory_max_size,self.dimMap, self.dimMap))
        self.dones = np.zeros((self.memory_max_size))
        self.previous_robot_data = np.zeros((self.memory_max_size, 4))
        self.next_robot_data = np.zeros((self.memory_max_size, 4))

        self.nmemory = 0





    def __str__(self):
        return "L'agent a pour coordonnées : "+str(self.position)

    #def update_target_model(self):
    #    self.target_network.set_weights(self.main_network.get_weights())
    """
    version continue

    def rotate_robot(self, angle_degrees):
        self.orientation = angle_degrees
        self.top_point = rotate_point(self.top_point, angle_degrees, self.position)
        self.bottom_right = rotate_point(self.bottom_right, angle_degrees, self.position)
        self.bottom_left = rotate_point(self.bottom_left, angle_degrees, self.position)
    def changer_vitesse(self, vitesse):
        if vitesse < 10 :
            self.vitesse = vitesse
    """
    
    def rotate_robot(self, direction):
        self.orientation += direction*18  # Si 0 -> -1 => -18 degrés     si 1 -> 0 => + 0    si 2 -> 1 => +18 degrés
        self.orientation %= 360

    def changer_vitesse(self, vitesse):
        self.vitesse = np.clip((vitesse*0.05+1)*self.vitesse, 1, 5)   # SI 0 -> -1 => vitesse = vitesse - 0.05*vitesse 
                                                    # si 2 -> 1 => vitesse = vitesse + 0.05*vitesse       si 1 -> 0 pas de changement 
        
            
    def avancer(self, agent_info):
        radian_angle = math.radians(self.orientation)
        if agent_info==0 :   # SI IL EST SUR DU ROUGE
            delta_x = (self.vitesse/4) * np.cos(radian_angle)  
            delta_y = (self.vitesse/4) * np.sin(radian_angle) 
        else :
            delta_x = self.vitesse * np.cos(radian_angle)  
            delta_y = self.vitesse * np.sin(radian_angle) 
        self.position += np.array([delta_x, delta_y], dtype=np.float32) 
        self.vagueAleatoire()
        self.position = np.nan_to_num(self.position, nan=0)

    def vagueAleatoire(self):
        proba = np.random.random()
        if proba < 0.4  :
            if proba<0.1:
                self.position += np.array([0, 1], dtype=np.float32)   
            elif proba<0.2:
                self.position += np.array([0, -1], dtype=np.float32) 
    
            elif proba<0.3:
                self.position += np.array([-1, 0], dtype=np.float32)
                
            else :
                self.position += np.array([1, 0], dtype=np.float32)
                


    def remember(self, transition):
      
        self.sampled_previous_states[self.nmemory] = transition[0]
        self.sampled_actions[self.nmemory] = transition[1]

        self.rewards [self.nmemory] = transition[2]
        self.next_states[self.nmemory] = transition[3]
        self.dones [self.nmemory] = transition[4]
        self.previous_robot_data[self.nmemory] = transition[5]
        self.next_robot_data [self.nmemory] = transition[6]



        self.nmemory=(self.nmemory+1)%self.memory_max_size
         

    def replay(self, batch_size):
        sampled_indices = np.random.choice(range(self.nmemory), size=batch_size)

        sampled_previous_states = self.sampled_previous_states[sampled_indices] 
        sampled_actions = self.sampled_actions[sampled_indices] 
        sampled_rewards = self.rewards[sampled_indices] 
        sampled_next_states = self.next_states[sampled_indices] 
        sampled_dones = self.dones[sampled_indices] 
        sampled_next_agents = self.next_robot_data[sampled_indices] 
        sampled_previous_agents = self.previous_robot_data[sampled_indices] 


        sampled_transitions = [sampled_previous_states, sampled_actions, sampled_rewards, 
                               sampled_next_states, sampled_dones, sampled_previous_agents, sampled_next_agents]
        return sampled_transitions
    
    def replace_agent(self, point):
        self.position = point

    def choose_action(self, state, training=True):
        #start_time = time.time()
        if (self.nmemory < 10 or self.epsilon > np.random.rand()) and training:
            action = np.random.randint(0, 9)    # NOUVELLE DEFINITION DES ACTIONS
                                                        # VITESSE : 0 dimininution 5% 1 constant 2 augmentation 5%
        else:                                           # ANGLE : 0 -pi/10  1 : constant 2 +pi/10
            input_data = tf.expand_dims(state, axis=0)
            robot_data = np.array([[self.position[0], self.position[1], self.vitesse, self.orientation]])
            actions = self.main_network([input_data, robot_data], training=False)
            maps = self.encoder([input_data, robot_data], training=False)
            maps = np.array(maps)
            for i in range(6):
                img = maps[0,:, :, i]
                print(img)
                img = np.clip(img * 255, 0, 255)
                img = img.astype(np.uint8)  
                name = 'pixel'+str(i)+'.png'
                image = Image.fromarray(img)
                image.save(name)
            
            action = np.argmax(actions)

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_factor)
        
        return action
    

    def load_model(self, model) :
        self.main_network = model
