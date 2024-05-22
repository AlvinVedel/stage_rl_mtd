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

var = 'k'

# %%
####### MODELE ######
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def create_model():
    print("creating model...")
    input_commun = layers.Input(shape=(400, 400, 7))  # Entrée pour les images
    conv_layer1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(6, 6), activation='relu')(input_commun)
    conv_layer2 = layers.Conv2D(64, kernel_size=(6, 6), strides=(4, 4), activation='relu')(conv_layer1)
    
    conv_layer3 = layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(conv_layer2)
    conv_layer4 = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu')(conv_layer3)
    pooling_layer2 = layers.MaxPooling2D(pool_size=(2, 2))(conv_layer4)
    flatten_layer = layers.Flatten()(pooling_layer2)
    layer = layers.Dense(64, activation='relu')(flatten_layer)
    layer2a = layers.Dense(64, activation='relu')(layer)
    layer2b = layers.Dense(64, activation='relu')(layer)


    angle_output = layers.Dense(4, activation='softmax', name='angle_output')(layer2a)
    speed_output = layers.Dense(4, activation='softmax', name='speed_output')(layer2b)

    model = tf.keras.Model(inputs=input_commun, outputs=[angle_output, speed_output])
    #model.compile(optimizer='Adadelta', loss={'angle_output': 'binary', 'speed_output': 'mean_squared_error'})
    print("model created")
    return model


# %%
"""
Principe :

environnement est l'image : on part de template, on place le robot avec redraw selon sa position   =>  state

On choisit une action :
    Au début c'est du hasard jusqu'à ce qu'on ait collecté suffisement de données
    Après on fait une prédiction à partir du modèle de deep pour maximiser la Q value

Transition : on passe d'un état s -> action a -> s+1

La transition s'accompagne d'un reward R 


Points importants :
    Expérience replay sert à vérifier qu'on effectue toujours les mêmes actions après des màj du réseau
    Réseau cible mis à jour toutes les 10 000 itérations avec les points du réseau principal
    On n'entraine pas le réseau principal à chaque itération (toutes les 4, ..., 10)
    On maintient une part de hasard dans le choix des action : rôle des epsillones


"""


# %%
####### VISUEL #####
def rotate_point(point, angle_degrees, centre):
        angle_radians = math.radians(angle_degrees)    
        translated_point = (point[0] - centre[0], point[1] - centre[1])
        rotated_x = translated_point[0] * math.cos(angle_radians) - translated_point[1] * math.sin(angle_radians)
        rotated_y = translated_point[0] * math.sin(angle_radians) + translated_point[1] * math.cos(angle_radians)
        
        # Déplacer le point de retour à sa position d'origine
        rotated_point = (rotated_x + centre[0], rotated_y + centre[1])
        return rotated_point

# %%
####### CLASSE ROBOT   #######

class MonRobot:
    def __init__(self, position, orientation=0, vitesse=0):
        self.position = position  # TUPLE (x, y)
        self.orientation = orientation  # en degree
        self.vitesse = vitesse # Nb de pixels à avancer par itération
        self.height = 20  # taille de la médiane issue du sommet principal
        self.width=10 # distance entre les 2 sommets opposés au principal
        self.top_point = (position[0],position[1]-(2/3)*self.height)
        self.bottom_right = (position[0]+self.width/2,  position[1]+self.height/3)
        self.bottom_left = (position[0]-self.width/2, position[1]+self.height/3)
        self.main_network = create_model()
        self.target_network = tf.keras.models.clone_model(self.main_network)
        self.memory = np.array([])
        self.memory_max_size = 10000
        self.step_count = 0
        self.epsilon = 1
        self.min_epsilon = 0.1
        self.epsilon_factor = 0.9

    def update_target_model(self):
        self.target_network.set_weights(self.main_network.get_weights())
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
        if direction == 0 :
            self.orientation = 0
        elif direction == 2 :
            self.orientation = 180
        elif direction == 1 :
            self.orientation = 90
        else :
            self.orientation = 270
        self.top_point = rotate_point(self.top_point, self.orientation, self.position)
        self.bottom_right = rotate_point(self.bottom_right, self.orientation, self.position)
        self.bottom_left = rotate_point(self.bottom_left, self.orientation, self.position)
    def changer_vitesse(self, vitesse):
            # les vitesse qui arrivent en entrée sont déjà 0, 1, 2, 3
        self.vitesse = vitesse
        

            
    def avancer(self):
        radian_angle = math.radians(self.orientation)  
        delta_x = self.vitesse * math.cos(radian_angle)  
        delta_y = self.vitesse * math.sin(radian_angle) 
        self.position = (self.position[0] + delta_x, self.position[1] + delta_y)

    def remember(self, transition):
        if len(self.memory) == 0:
            self.memory = np.array([transition], dtype=object)
        else:
            self.memory = np.roll(self.memory, 1, axis=0)
            self.memory[0] = transition
        if len(self.memory) > self.memory_max_size:
            self.memory = self.memory[:self.memory_max_size]

    def replay(self, batch_size):
        sampled_transitions = np.random.choice(self.memory, size=batch_size)
        return sampled_transitions
    
    def replace_agent(self, point):
        self.position = point

    def choose_action(self, state):
        if len(self.memory) < 50000 and self.epsilon > np.random.rand():
            direction = np.random.randint(0, 4)
            vitesse = np.random.randint(0, 4)
            action = (direction, vitesse)
        else:
            actions_probs = self.main_network(state)
            direction = keras.ops.argmax(actions_probs[0]).numpy()
            vitesse = keras.ops.argmax(actions_probs[1]).numpy()
            action = (direction, vitesse)  

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_factor)
       
        return action

# %%
####### Classe environnement  ######

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def draw_basic_map() :
    print("in draw")
    largeur = 400
    hauteur = 400
    image = Image.new("RGB", (largeur, hauteur), color=(255, 0, 0))
    draw = ImageDraw.Draw(image)

    points_controle1 = [(120, 400), (80, 330), (70, 250), (140, 200), (180, 100), (270, 0) ]
    draw.line(points_controle1, fill=(0, 255, 0), width=2)

    points_controle2 = [(210, 400), (120, 330), (110, 250), (190, 200), (225, 100), (330, 0) ]
    draw.line(points_controle2, fill=(0, 255, 0), width=2)

    all_points = points_controle1 + points_controle2[::-1]  # Inverser la deuxième liste de points
    draw.polygon(all_points, fill=(0, 255, 0))
    cercle_centre = (280, 30)
    rayon = 7
    draw.ellipse((cercle_centre[0] - rayon, cercle_centre[1] - rayon,
               cercle_centre[0] + rayon, cercle_centre[1] + rayon), fill=(255, 255, 255))
    #draw.point(cercle, fill=(255, 255, 255))
    return image



class Environnement:
    def __init__(self):
        self.base = draw_basic_map()
        self.visuel = self.base.copy()
        self.agent = None
        self.state = None    
        self.previous_distance = None  
        #self.objectif_position = objectif
    def append_agent(self, robot):
        self.agent=robot
        self.redraw()
        self.encode_map()
        indices_obj = np.argwhere(self.state[:, :, 3] == 1)
        distances = np.linalg.norm(indices_obj - robot.position, axis=1)
        distance = np.mean(distances)
        self.previous_distance = distance
        
    def redraw(self):
        if self.agent != None and self.visuel!=None :
            draw = ImageDraw.Draw(self.visuel)
            triangle_points = [self.agent.top_point, self.agent.bottom_right, self.agent.bottom_left]
            draw.polygon(triangle_points, fill=(0, 0, 0))
            
    def encode_map(self) :
        if self.visuel == None or self.agent == None:
            print('required components to encode are missing')
        else :
            img_array = np.array(self.visuel)
            colors = {
                (255, 0, 0): [1, 0, 0, 0, 0, 0, 0],   # Rouge: Mur
                (0, 255, 0): [0, 1, 0, 0, 0, 0, 0],   # Vert: Chemin
                (0, 0, 0): [0, 0, 1, 0, 0, 0, 0],     # Noir: Robot
                (255, 255, 255): [0, 0, 0, 1, 0, 0, 0]  # Blanc: Objectif
            }
            # Créer un masque booléen pour chaque couleur
            masks = np.zeros((img_array.shape[0], img_array.shape[1], len(colors)), dtype=bool)
            for idx, color in enumerate(colors):
                masks[:, :, idx] = np.all(img_array == np.array(color), axis=-1)
            # Utiliser les masques pour remplacer les valeurs
            processed_image = np.zeros((img_array.shape[0], img_array.shape[1], 7))
            for idx, color in enumerate(colors.values()):
                processed_image += np.where(masks[:, :, idx][:, :, np.newaxis], color, 0)
            self.state = processed_image


    def compute_new_state(self):
        self.visuel = self.base.copy()
        self.append_agent(self.agent)
        self.state = self.encode_map()

    def reset_env(self):
        self.visuel = draw_basic_map()
        self.agent = None
        self.previous_distance = None
        self.state = None
        return self

    def compute_reward(self):
        robot_point = self.agent.position
        agent_info = self.state[int(robot_point[0]), int(robot_point[1])]

        indices_obj = np.argwhere(self.state[:, :, 3] == 1)
        distances = np.linalg.norm(indices_obj - robot_point, axis=1)
        distance = np.mean(distances)
        difference = self.previous_distance - distance
        reward = 0
        
        reward += difference
        # Si première composante de 3ème dimension = 1 : rouge alors on pénalise
        if agent_info[0] == 1:
            reward -= 5  

        self.previous_distance = distance
        return reward
       
    
    def step(self, action):
        # Action est un tuple : action 0 = rotation  action 1 = vitesse   (premier exemple avec 4 modalités chacune)
        previous_state = self.state

        self.agent.rotate_robot(action[0])
        self.agent.changer_vitesse(action[1])
        self.agent.avancer()
        reward = self.compute_reward()

        new_state = self.compute_new_state()
        if self.previous_distance >=10 :
            done=1
        else :
            done=0

        episode = (previous_state, action, reward, new_state, done)
        self.agent.remember(episode)

        return new_state, reward, done
    
#image = draw_basic_map()
#image.save("./projet_1/test.png")



# %%
nb_ep = 1
nb_steps = 40
point_depart = (150, 390)
gamma = 0.99
loss_function = keras.losses.SparseCategoricalCrossentropy()

optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)


# %%

env = Environnement()
print("env created")
robot = MonRobot(point_depart)  # Un peu long, environ 3mn ?
print("robot created")


# %%


all_frames = []
nb_eps =0

for episode in range(nb_ep) :
    nb_eps+=1
    print("épisode : "+str(nb_ep))
    env.reset_env()
    env.append_agent(robot)
    nb_frames = 0
    total_reward = 0

    episode_frames = []

    for step in range(nb_steps) :
        nb_frames+=1
        print("step ", str(nb_frames))

        action = env.agent.choose_action(env.state)
        # La greedy policy est définie dans la méthode choose action 

        new_state, reward, done = env.step(action)
        # la fonction remember est déjà appelée dans step

        if nb_frames % 4 == 0 and len(env.agent.memory) > 32 :
            print("c'est une frame 4")

            rdm_tran = env.agent.replay(32) 
            states, actions, rewards, next_states, dones = zip(*rdm_tran)
            # On pioche dans la mémoire 32 éléments, tuple de 5 mtn on a ajouté done
    
            states = np.array(states)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            actions = np.array(actions)
            dones = np.array(dones)


            future_rewards = env.agent.target_network.predict(next_states)
            # Q value = reward + discount factor * expected future reward : comme dans DQN keras
            updated_q_values = rewards + gamma * keras.ops.amax(
                future_rewards, axis=1
            )
            updated_q_values = updated_q_values * (1 - dones) - dones

            # 2 masques pour chacune de mes variables
            action_mask_angle = tf.one_hot([a[0] for a in actions], 4)
            action_mask_speed = tf.one_hot([a[1] for a in actions], 4)

            with tf.GradientTape() as tape:
                # Prédiction des Q-values pour les états actuels échantillonnés
                q_values_angle, q_values_speed = env.agent.main_network(states)

                # Sélection des Q-values pour les actions prises
                q_action_angle = tf.reduce_sum(q_values_angle * action_mask_angle, axis=1)
                q_action_speed = tf.reduce_sum(q_values_speed * action_mask_speed, axis=1)

                # Calcul de la perte moyenne pour chaque sortie
                loss_angle = tf.reduce_mean(loss_function(updated_q_values, q_action_angle))
                loss_speed = tf.reduce_mean(loss_function(updated_q_values, q_action_speed))
                # Perte totale
                loss = loss_angle + loss_speed

            # Calcul du gradient et application des mises à jour des poids
            grads = tape.gradient(loss, env.agent.main_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, env.agent.main_network.trainable_variables))


        if nb_frames % 10000 == 0 :
            env.agent.update_target_model()
        """
        if done :
            break
        """
        episode_frames.append(env.visuel)

    if episode_frames:
        name = "./animation"+str(nb_ep)+'.gif'
        episode_frames[0].save(name, save_all=True, append_images=episode_frames[1:], duration=10*len(episode_frames), loop=0)
    else:
        print("No frames captured, unable to save animation.")
        





# %%


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
