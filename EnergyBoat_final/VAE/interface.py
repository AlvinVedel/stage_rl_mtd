import numpy as np
from EnergyBoatScenario.utils_env import *
from EnergyBoatScenario.envNED import EnergyBoatEnv
import pygame
from EnergyBoatScenario.pygame_utils import Slider, ScrollableWindow
import sys
import math
import tensorflow as tf
from keras import layers
#import keras
from keras.models import load_model


directory = './EnergyBoatScenario/'
track_parameters_file = 'scenario_parameters.yaml'

class Sampling(layers.Layer):
    """Uses (mu, logvar) to sample z, the vector encoding an observation."""
    def call(self, inputs):
        mu, logvar = inputs
        batch = tf.shape(mu)[0]
        dim = tf.shape(mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mu + tf.exp(0.5 * logvar) * epsilon

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config

class VariationalLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(VariationalLayer, self).__init__(**kwargs)
        self.dense_mu = layers.Dense(512)
        self.dense_logvar = layers.Dense(512)
        self.sampling = Sampling()

    def call(self, inputs):
        mu = self.dense_mu(inputs)
        logvar = self.dense_logvar(inputs)
        z = self.sampling((mu, logvar))
        self.add_loss(-0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar)))
        return z

    def get_config(self):
        config = super(VariationalLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# A AJOUTER POUR CLONE ET AVANT DE LOAD LE MODEL
tf.keras.utils.get_custom_objects().update({
    'VariationalLayer': VariationalLayer,
    'Sampling': Sampling
})

env_config={
        "render_flag": True,
        "activate_pilot_behavior": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'activate_pilot_behavior'),
        "monaco_map":directory + 'assets/monaco_map_bgd.png',
        "font":directory + 'assets/font/arial.ttf',
        "implementation":"simple",
        "critical_battery_level": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'critical_battery_level'),
        "sufficient_battery_level": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'sufficient_battery_level'),
        "minimal_required_laps": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'minimal_required_laps'),
        "time_scaling_factor": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'time_scaling_factor'),
        "max_race_time": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_race_time'),
        "max_loop_track_time": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_loop_track_time'),
        "base_consumption_rate": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'base_consumption_rate'),
        "speed_consumption_factor": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'speed_consumption_factor'),
        "rayon_bouee": load_values_from_yaml(track_parameters_file, 'track_drawings', 'rayon_bouee'),
        "bouees": load_bouees_from_yaml(track_parameters_file, 'track_drawings', 'bouees'),
        "big_hexagon": load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'big_hexagon'),
        "small_hexagon": load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'small_hexagon'),
        "starting_zone_polygon":load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'starting_polygon'),
        "quadrilaterals":load_quadrilaterals_from_yaml(track_parameters_file, 'track_drawings', 'quadrilaterals'),
        "random_starting_zone_one": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'random_starting_zone_one'),
        "random_starting_zone_two": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'random_starting_zone_two'),
        "random_starting_zone_three": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'random_starting_zone_three'),
        "detection_ranges": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detection_ranges'), 
        "detection_ranges_for_display": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detection_ranges_for_display'), 
        "detector_angle_of_view": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detector_angle_of_view'), 
        "nb_detection_sectors": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'nb_detection_sectors'),
        "distance_ref_px" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'distance_ref_px'),
        "distance_ref_m" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'distance_ref_m'),
        "detector_values" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detector_values'),
        "directions": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'directions'),
        "speeds": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'speeds'),
        "concurrents_speeds": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'concurrents_speeds'),        
        "frames_per_second": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'frames_per_second'),
        "earth_radius": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'earth_radius'),
        "lat_lon_point_ref_one": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'lat_lon_point_ref_one'),
        "image_point_ref_one": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'image_point_ref_one'),
        "lat_lon_point_ref_two": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'lat_lon_point_ref_two'),
        "image_point_ref_two": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'image_point_ref_two'),
        }

env = EnergyBoatEnv(env_config)
s0 = env.reset()  # [cones, sups]
model = load_model("./speeded_up_endu_model2_6_VAE2_512_WDIST.h5")
latent_dim = 512


for layer in model.layers:
    print(layer.name)

latent_model = tf.keras.Model(model.input, model.get_layer('variational_layer').output)

latent_input = layers.Input(shape=(latent_dim,))  # Supposons que la dimension latente est 128

x = model.get_layer('dense_2')(latent_input)  # Connexion à la première couche Dense après la couche latente
x = model.get_layer('dense_3')(x)  # Connexion à la deuxième couche Dense après la couche latente
output = model.get_layer('global_output')(x)  # Connexion à la couche de sortie

# Créez un modèle de prédiction
prediction_model = tf.keras.Model(inputs=latent_input, outputs=output)


 

actionSet = {
    0: [-1.5, -15],
    1: [-1.5, -6],
    2: [-1.5, 0],
    3: [-1.5, 6],
    4: [-1.5, 15],
    5: [-0.6, -15],
    6: [-0.6, -6],
    7: [-0.6, 0],
    8: [-0.6, 6],
    9: [-0.6, 15],
    10: [0, -15],
    11: [0, -6],
    12: [0, 0],
    13: [0, 6],
    14: [0, 15],
    15: [0.6, -15],
    16: [0.6, -6],
    17: [0.6, 0],
    18: [0.6, 6],
    19: [0.6, 15],
    20: [1.5, -15],
    21: [1.5, -6],
    22: [1.5, 0],
    23: [1.5, 6],
    24: [1.5, 15]
}

pygame.init()
colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'purple', 'orange']

# Dimensions de la fenêtre
window_size = (1500, 720)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption('Environnement')

#background_image = pygame.image.load('background_image.jpg')
#background_image = pygame.transform.scale(background_image, window_size)

ROUGE = (255, 0, 0)
BLEU = (0, 0, 255)
VERT = (0, 255, 0)

BLANC = (255, 255, 255)
GRIS = (127.5, 127.5, 127.5)
NOIR  = (0, 0, 0)
GRIS_CLAIR = (200, 200, 200)



def draw_menu(screen, slider):
    pygame.draw.rect(screen, GRIS_CLAIR, (1200, 0, 300, 600))  # Menu sur la droite
    

    slider.draw(screen)

    min_value_text = small_font.render(f'{slider.min_value}', True, NOIR)
    max_value_text = small_font.render(f'{slider.max_value}', True, NOIR)
    
    screen.blit(min_value_text, (slider.rect.x , slider.rect.y - 10))
    screen.blit(max_value_text, (slider.rect.x + slider.rect.width -10, slider.rect.y - 10))
    

def draw_q_values(screen, q_values, position, size, titre, labels):
    max_q = max(q_values)
    min_q = min(q_values)
    
    label_surface = font.render(titre, True, NOIR)
    screen.blit(label_surface, (position[0], position[1] - 40))

    for i, q in enumerate(q_values):
        # Normaliser la valeur Q pour qu'elle se situe entre 0 et 255
        normalized_value = int(255 * (q - min_q) / (max_q - min_q))
        color = (255 - normalized_value, 255 - normalized_value, 255)
        
        x = position[0] + i * size
        y = position[1]
        
        pygame.draw.rect(screen, color, (x, y, size, size))

        value_surface = font.render(str(labels[i]), True, NOIR)
        screen.blit(value_surface, (x + 5, y + 5))

def draw_bar_charts(screen, data, nb_obs, titre):
    # data de shape  N, 25  => on fait une somme axe 0 pour obtenir 25,   et on divise par nb_obs pour les fréquences
    data = np.sum(data, axis=0)
    data = data / nb_obs  # fréquences entre 0 et 1

    for i, value in enumerate(data):
        pygame.draw.rect(screen, NOIR, (1220 + i * 10, 100- int(value * 200), 10, int(value * 200)))

    text = font.render(titre, True, NOIR)
    text_rect = text.get_rect(center=(1320, 120))
    screen.blit(text, text_rect)

def draw_q_bar_charts(screen, data, nb_obs, x, y, titre):
    # cette fois data de shape N, 5  on fait moyenne sur axe 0 pour obtenir 5, 
    data = np.mean(data, axis=0)  # il faut quand meme normaliser entre 0 et 1
    data = data - np.min(data)
    data = data /np.sum(data)

    #data = data / nb_obs  # fréquences entre 0 et 1

    for i, value in enumerate(data):
        pygame.draw.rect(screen, NOIR, (x + i * 10, y - int(value * 200), 10, int(value * 200)))
    text = font.render(titre, True, NOIR)
    text_rect = text.get_rect(center=(x + 5, y + 20))
    screen.blit(text, text_rect)









vitesse_labels = ["-1.5", "-0.6", "0", '+0.6', '+1.5']
direction_labels = ["-15°", "-6°", "0°", "+6°", "+15°"]
font = pygame.font.Font(None, 30)
small_font = pygame.font.Font(None, 15)
#slider1 = 
sliders = [Slider(1270, 330+i*50, 150, 20, -3, 3, 0, i+1) for i in range(latent_dim)]
SCROLL_WINDOW_WIDTH = 300
SCROLL_WINDOW_HEIGHT = 600

scrollable_window = ScrollableWindow(1200, 300, SCROLL_WINDOW_WIDTH, SCROLL_WINDOW_HEIGHT, sliders)

affichage_classique = True
memory_size = 1000
memory_count = 0
memory_index = 0
memory_latent = np.zeros((memory_size, latent_dim))
memory_action = np.zeros((memory_size, 25))
memory_vitesse = np.zeros((memory_size, 5))
memory_direction = np.zeros((memory_size, 5))

while True  :

    sliders_values = tf.tile(tf.expand_dims(tf.convert_to_tensor(np.array([slider.value for slider in sliders]), dtype=tf.float32), axis=0), [env.nb_agents, 1]) # shape 4, 256
    
    latent_probs = latent_model(s0, training=False)
    new_latent = latent_probs+sliders_values  # shape 4, 256

    #print(latent_probs.shape)
    q_vals = prediction_model(new_latent, training=False).numpy()
    #q_vals = model(s0, training=False).numpy()  # 4, 25
    #print(q_vals)
    #q_vals = np.random.randint(-20, 20, (4, 25)) * np.random.random((4, 25))
    action = np.argmax(q_vals, axis=1)
    actions = np.zeros((env.nb_agents, 2))
    
    q_mat = q_vals.reshape((4, 5, 5))
    q_vitesse = np.sum(q_mat, axis=2)
    q_angle = np.sum(q_mat, axis=1)

    #print(q_vals[0])
    #print(q_mat)
    #print(q_vitesse)
    #print(q_angle)

    for i in range(env.nb_agents) :
        actions[i] = actionSet[action[i]]
        memory_latent[memory_index] = latent_probs[i].numpy()
        null_vec = np.zeros(25)
        null_vec[action[i]] = 1  # vecteur 25, avec un 1 à l'emplacement de l'action
        memory_action[memory_index] = null_vec
        memory_vitesse[memory_index] = q_vitesse[i]
        memory_direction[memory_index] = q_angle[i]
        memory_index = (memory_index+1)%memory_size
        memory_count = min(memory_count+1, memory_size)
        


    

    
    s0, r, d, dd, u1, u2 = env.step(actions)
    if np.all(d) :
        s0 = env.reset()
    

    pygame.display.set_caption("Monaco Energy boat demo")
    
               
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                s0 = env.reset()
            elif event.key == pygame.K_d :
                affichage_classique = not affichage_classique


        for slider in sliders:
            slider_value = slider.handle_event(event)
            if slider_value is not None:
                print("Slider "+str(slider.id)+" value : "+str(slider_value))

        scrollable_window.handle_event(event)

    scrollable_window.update()




                                
    bg = pygame.image.load(env.monaco_map).convert_alpha()
            
    screen.blit(bg, (0, 0))

    
    for i, bouee in enumerate(env.bouees) :
        circle_center = (bouee[0], bouee[1])
        pygame.draw.circle(screen, ORANGE, circle_center, env.rayon_bouee)

    for i, gate in enumerate(env.current_gateway) :
        circle_center = (gate[0], gate[1])
        pygame.draw.circle(screen, colors[i], circle_center, env.rayon_gateways[env.current_gateway_index[i]])
            
   # draw_bouees(screen, env.bouees, env.rayon_bouee)   
    big_hexagon = [(point[0], point[1]) for point in env.big_hexagon]
    small_hexagon = [(point[0], point[1]) for point in env.small_hexagon]
    start_zone = [(point[0], point[1]) for point in env.starting_zone_polygon]
    pygame.draw.polygon(screen, ROUGE, big_hexagon, 1)   
    pygame.draw.polygon(screen, ROUGE, small_hexagon, 1)  
    pygame.draw.polygon(screen, ROUGE, start_zone, 1)  

    for i, agent_pos in enumerate(env.agent_positions) :
        if i == env.followed :
            pygame.draw.circle(screen, colors[i], (agent_pos[0], agent_pos[1]), 10, width=1)
        else :
            pygame.draw.circle(screen, colors[i], (agent_pos[0], agent_pos[1]), 10, width=1)
            

    detection_angle = 110
    detection_range = 40
    angle_step = 12  # Pas d'angle entre chaque segment
    distance_step = 10  # Pas de distance pour chaque segment

    for i in range(len(env.agent_positions)):
        if not env.dones[i]:
            color = colors[i]
            boat_orientation = np.rad2deg(env.agent_directions[i])
            start_angle = boat_orientation - detection_angle / 2
            end_angle = boat_orientation + detection_angle / 2
            
            while start_angle < end_angle:
                actual_angle = start_angle + angle_step
                for j in range(distance_step, detection_range + 1, distance_step):
                    start_angle_rad = np.deg2rad(start_angle)
                    end_angle_rad = np.deg2rad(actual_angle)
                    
                    start_x = env.agent_positions[i][0] + j * math.cos(start_angle_rad)
                    start_y = env.agent_positions[i][1] + j * math.sin(start_angle_rad)
                    end_x = env.agent_positions[i][0] + j * math.cos(end_angle_rad)
                    end_y = env.agent_positions[i][1] + j * math.sin(end_angle_rad)
                    
                    # Dessiner le cône de détection autour du bateau
                    pygame.draw.polygon(screen, color, [
                        (env.agent_positions[i][0], env.agent_positions[i][1]), 
                        (start_x, start_y), 
                        (end_x, end_y)
                    ], 1)
                    
                start_angle += angle_step
            
    pygame.draw.rect(screen, BLANC, (1200, 0, 300, 300)) 
    #draw_detection_field(draw, self.agent_positions, self.agent_directions, self.dones, colors)
    scrollable_window.draw(screen)
    #draw_menu(screen, slider1) 
    
    if affichage_classique : 
        draw_q_values(screen, q_angle[env.followed], (1240, 50), 50, "DIRECTIONS", direction_labels)
        draw_q_values(screen, q_vitesse[env.followed], (1240, 150), 50, "VITESSES", vitesse_labels)
    else :
        draw_bar_charts(screen, memory_action, memory_count, "Actions")
        draw_q_bar_charts(screen, memory_vitesse, memory_count, 1250, 250, "Vitesse")
        draw_q_bar_charts(screen, memory_direction, memory_count, 1400, 250, "Direction")

    

    pygame.display.flip()  # Mettre à jour l'affichage     
    pygame.time.wait(100)
    #pygame.time.Clock().tick(60)

    





    #env.render()

# Couleurs
WHITE = (255, 255, 255)