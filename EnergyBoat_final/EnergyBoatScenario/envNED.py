from copy import deepcopy
import yaml
import time
import sys
import math
import random
import time
import pygame
import numpy as np
from pygame.locals import QUIT
import gymnasium as gym
from PIL import Image, ImageDraw


from EnergyBoatScenario.bridge import *
from EnergyBoatScenario.utils_env import *
from EnergyBoatScenario.view2d import *



colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'purple', 'orange']


def area_triangle(point1, point2, point3):
    return 0.5 * np.abs((point1[0] - point3[0]) * (point2[1] - point1[1]) - (point1[0] - point2[0]) * (point3[1] - point1[1]))

def area_quadrilataire(p1, p2, p3, p4):
    a1 = area_triangle(p1, p2, p3)
    a2 = area_triangle(p3, p4, p1)
    return a1+a2

def is_inside_hexagon(vertices, point):
    """
    Détermine si un point est à l'intérieur d'un polygone convexe ou non convexe.
    
    Arguments:
    point -- Coordonnées du point à tester (x, y).
    vertices -- Liste des sommets du polygone [(x1, y1), (x2, y2), ...].
    
    Retourne:
    True si le point est à l'intérieur du polygone, False sinon.
    """
    x, y = point
    n = len(vertices)
    inside = False

    px, py = vertices[0]
    for i in range(1, n + 1):
        vx, vy = vertices[i % n]
        if ((py > y) != (vy > y)) and (x < (vx - px) * (y - py) / (vy - py) + px):
            inside = not inside
        px, py = vx, vy

    return inside



def generate_random_point_in_quadrilateral(p1, p2, p3, p4):
    """Genere un point aleatoire dans un quadrilatere definit par 4 points"""
    r1 = random.random()
    r2 = random.random()
    
    x = (1 - r1) * (1 - r2) * p1[0] + r1 * (1 - r2) * p2[0] + r1 * r2 * p3[0] + (1 - r1) * r2 * p4[0]
    y = (1 - r1) * (1 - r2) * p1[1] + r1 * (1 - r2) * p2[1] + r1 * r2 * p3[1] + (1 - r1) * r2 * p4[1]
    
    return np.array([x, y])

def generate_points_in_quadrilateral(sz, nb_points, dist):
      """
      Genere nb_points dans une zone de depart sz tq les points sont à une distance dist les uns des autres
      """
      diag = np.eye(nb_points)*dist**3

      while True : 
        points = np.zeros((nb_points, 2))  # 4, 2  => 1, 4, 2
        for i in range(nb_points) :
          points[i] = generate_random_point_in_quadrilateral(sz[0], sz[1], sz[2], sz[3])
        distance = np.sum((np.expand_dims(points, axis=0) - np.expand_dims(points, axis=1))**2, axis=2)   # 1, 4, 2  et 4, 1, 2  ==> 4, 4, 2 puis somme axe 2 => 4, 4
        nd = distance + diag 
        if np.all(nd >= dist**2):
          return points


def generate_circuit(seed=42, pixels_per_meter=1, a=0, nb_agents=4, distance_agents=30) :
    """
    Fonction de creation de circuits aléatoires, utilisation de seed pour la reproductibilité des résultats
    Génère les points et formes géométriques dans un repère pixel puis les convertis en NED grace au pixels_per_meter
    """

    random.seed(seed)
    np.random.seed(seed+10)
    a = math.sqrt(2*a)
    max_x = 1280
    max_y = 720
    # ON DIVISE L IMAGE EN 6 ZONES : 2 DANS LE SENS DE LA HAUTEUR ET 3 DANS LA LARGEUR
    # ON TIRE LES 6 POINTS DE L HEXAGON EXTERIEUR ET ON DEFINIT UN CIRCUIT DE 50 DE LARGEUR
    x1 = random.randint(0+20, int((max_x/3))-50)
    y1 = random.randint(0+20, int(max_y/2)-100)
    p1 = [x1, y1]

    x2 = random.randint(int((max_x/3))+50, int((2*max_x/3))-50)
    y2 = random.randint(0+20, int(max_y/2)-150)
    p2 = [x2, y2]

    x3 = random.randint(int((2*max_x/3))-50, max_x-20)
    y3 = random.randint(0+20, int(max_y/2)-100)
    p3 = [x3, y3]

    x4 = random.randint(int((2*max_x/3))-50, max_x-20)
    y4 = random.randint(int(max_y/2)+100, max_y-20)
    p4 = [x4, y4]

    x5 = random.randint(int((max_x/3))+50, int((2*max_x/3))-50)
    y5 = random.randint(int(max_y/2)+150, max_y-20)
    p5 =[x5, y5]

    x6 = random.randint(0+20, int((max_x/3))-50)
    y6 = random.randint(int(max_y/2)+100, max_y-20)
    p6 = [x6, y6]

    big_hexagon = np.array([p1, p2, p3, p4, p5, p6])

    ## CALCUL DES BISSECTRICES :
    directions_bissectrices = np.zeros(6)

    dirb1_1 = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    dirb1_2 = np.arctan2(p6[1] - p1[1], p6[0] - p1[0])
    dirb1 = (dirb1_1+dirb1_2)/2
    directions_bissectrices[0] = dirb1

    dirb2_1 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])
    dirb2_2 = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    dirb2 = (dirb2_1+dirb2_2)/2 
    if dirb2 < 0 :
      dirb2 += np.pi

    directions_bissectrices[1] = dirb2

    dirb3_1 = np.arctan2(p4[1] - p3[1], p4[0] - p3[0])
    dirb3_2 = np.arctan2(p2[1] - p3[1], p2[0] - p3[0])
    dirb3 = (dirb3_1+dirb3_2)/2 
    if dirb3 < 0 :
      dirb3 += np.pi
    directions_bissectrices[2] = dirb3

    dirb4_1 = np.arctan2(p5[1] - p4[1], p5[0] - p4[0])
    dirb4_2 = np.arctan2(p3[1] - p4[1], p3[0] - p4[0])
    dirb4 = (dirb4_1+dirb4_2)/2 
    if dirb4 < np.pi and dirb4 > 0 :
      dirb4 -= np.pi
    directions_bissectrices[3] = dirb4

    dirb5_1 = np.arctan2(p6[1] - p5[1], p6[0] - p5[0])
    dirb5_2 = np.arctan2(p4[1] - p5[1], p4[0] - p5[0])
    dirb5 = (dirb5_1+dirb5_2)/2
    if dirb5 < np.pi and dirb5>0 :
      dirb5 -= np.pi
    directions_bissectrices[4] = dirb5

    dirb6_1 = np.arctan2(p1[1] - p6[1], p1[0] - p6[0])
    dirb6_2 = np.arctan2(p5[1] - p6[1], p5[0] - p6[0])
    dirb6 = (dirb6_1+dirb6_2)/2
    if dirb6 < 0 :
      dirb6 += 2*np.pi
    directions_bissectrices[5] = dirb6

    espace = 100
    small_hexagon = np.concatenate([np.expand_dims(big_hexagon[:, 0]+espace*np.cos(directions_bissectrices), axis=1), np.expand_dims(big_hexagon[:, 1]+espace*np.sin(directions_bissectrices), axis=1)], axis=1)
    bouees =  np.concatenate([np.expand_dims(big_hexagon[:, 0]+(espace+23)*np.cos(directions_bissectrices), axis=1), np.expand_dims(big_hexagon[:, 1]+(espace+23)*np.sin(directions_bissectrices), axis=1)], axis=1)

    random_start_zone = random.choice([2, 3, 5, 0]) # LES 4 COINS DE LA CARTE

    big_hexagon = np.concatenate([big_hexagon[random_start_zone:], big_hexagon[:random_start_zone]], axis=0)
    small_hexagon = np.concatenate([small_hexagon[random_start_zone:], small_hexagon[:random_start_zone]], axis=0)


    vector = np.array(small_hexagon[0]) - np.array(big_hexagon[0])
    # Trouver un vecteur orthogonal
    orthogonal = np.array([-vector[1], vector[0]])
    unit_vector = orthogonal / np.linalg.norm(orthogonal)
    scaled_vector = unit_vector * 300
    sz1 = np.array(big_hexagon[0]) + scaled_vector
    sz2 = np.array(small_hexagon[0]) + scaled_vector

    start_zone = np.array([sz1, big_hexagon[0], small_hexagon[0], sz2])


    init_pos = generate_points_in_quadrilateral(start_zone, nb_agents, distance_agents)
   
    small_hexagon = image_to_ned(small_hexagon[:, 0], small_hexagon[:, 1], pixels_per_meter)
    big_hexagon = image_to_ned(big_hexagon[:, 0], big_hexagon[:, 1], pixels_per_meter)
    start_zone = image_to_ned(start_zone[:, 0], start_zone[:, 1], pixels_per_meter)
    bouees = image_to_ned(bouees[:, 0], bouees[:, 1], pixels_per_meter)
    init_pos = image_to_ned(init_pos[:, 0], init_pos[:, 1], pixels_per_meter)

    return small_hexagon, big_hexagon, start_zone, bouees, init_pos




def image_to_ned(x, y, pixels_per_meter, x_origin=0, y_origin=0):
    """
    re implementation de la fonction image_to_ned qui peut prendre en entrée des np array et retourne des np array
    """
    delta_x = x - x_origin
    delta_y = y - y_origin
    north_y = -delta_y / pixels_per_meter
    east_x = delta_x / pixels_per_meter
    if isinstance(x, int) or isinstance(x, float):
      return np.array([east_x, north_y])
    else :
      return np.concatenate([np.expand_dims(east_x, axis=1), np.expand_dims(north_y, axis=1)], axis=1)
    
def ned_to_image(east_x, north_y, pixels_per_meter, image_width=1280, image_height=720):
    """
    re implementation de ned_to_image qui prend en entrée des np array et retourne des np array
    """
    x_origin = 0
    y_origin = 0
    x = east_x * pixels_per_meter + x_origin
    y = -north_y * pixels_per_meter + y_origin
    if isinstance(east_x, int) or isinstance(east_x, float) :
      return np.array([x, y]).astype(np.int32)
    else :
      return np.concatenate([np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)], axis=1).astype(np.int32)

def tuple_polygon_to_array(polygon):
    """
    fonction qui convertis des polygones tuples en matrices numpy (pour charger les données du yaml)
    """
    array_polygon = np.zeros((len(polygon), 2))
    for i in range(len(polygon)) :
        array_polygon[i] = np.array([polygon[i][0], polygon[i][1]])
    return array_polygon

class EnergyBoatEnv(gym.Env) : 
    """
    classe de l'environnement : 
    fonction init pour la création, reset pour ramener à un état s0
    get_env_state pour récupérer l'observation de l'environnement
    step pour appliquer une action à l'environnement
    render pour obtenir un retour graphique de l'environnement
    """

    def __init__(self, env_config, nb_agents=4, battery_consumption='medium', max_steps=-1, init_seed=42) :
        
        # CHARGEMENT DE CERTAINS PARAMETRES DU YAML
        self.font = env_config["font"]
        self.monaco_map = env_config["monaco_map"]
        self.distance_ref_px = env_config["distance_ref_px"] 
        self.distance_ref_m = env_config["distance_ref_m"]
        self.pixels_per_meter = self.distance_ref_px / self.distance_ref_m

        # RECUPERATION ET STOCKAGE DU VRAI CIRCUIT DE LA COURSE
        self.real_big_h = tuple_polygon_to_array(env_config["big_hexagon"])
        self.real_big_h = image_to_ned(self.real_big_h[:, 0], self.real_big_h[:, 1], self.pixels_per_meter)
        self.real_small_h = tuple_polygon_to_array(env_config["small_hexagon"])
        self.real_small_h = image_to_ned(self.real_small_h[:, 0], self.real_small_h[:, 1], self.pixels_per_meter)

        self.real_start_zone = tuple_polygon_to_array(env_config["starting_zone_polygon"])
        self.real_init_pos = generate_points_in_quadrilateral(self.real_start_zone, nb_agents, 30)
        self.real_start_zone = image_to_ned(self.real_start_zone[:, 0], self.real_start_zone[:, 1], self.pixels_per_meter)
        self.real_init_pos = image_to_ned(self.real_init_pos[:, 0], self.real_init_pos[:, 1], self.pixels_per_meter)

        self.real_rayon_bouee = env_config["rayon_bouee"]
        self.real_bouees = tuple_polygon_to_array(env_config["bouees"])
        self.real_big_h = image_to_ned(self.real_bouees[:, 0], self.real_bouees[:, 1], self.pixels_per_meter)
        
        self.screen_width_ned = SCREEN_WIDTH/self.pixels_per_meter  # IMPORTANT
        self.screen_height_ned = SCREEN_HEIGHT/self.pixels_per_meter   # SIGNE - car NED inverse les coordonnées y dans le clip
        
        self.render_flag= True
        

        ########### ATTRIBUTS AJOUTES #########

        self.nb_agents = nb_agents
        self.init_seed = init_seed


        random.seed(self.init_seed)
        # Générer une liste de seeds
        self.num_seeds = 10000
        self.seeds = [random.randint(0, 2**32 - 1) for _ in range(self.num_seeds)]
        self.maps = []
        self.maps_index = 0
        self.current_max_seeds = 0
        self.nb_maps = 10
        self.map_counter = 0
        # generation de cartes
        for i in range(self.nb_maps) :
            small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos = generate_circuit(seed = self.seeds[i], pixels_per_meter=self.pixels_per_meter)
            self.maps.append([small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]) # LE PREMIER 0 EST UN BOOLEEN QUI DIT SI LA MAP A ETE COMPLETE, LE 2EME POUR LE NOMBRE DE TRY SUR LA MAP
            self.current_max_seeds += 1

             
        self.followed = 0 # attribut qui contient le numéro du robot à afficher dans le render

        
        # définition de différent profils de batterie à l'aide de coefficients 
        self.battery_consumption = battery_consumption
        if self.battery_consumption == 'hard' :  # environ 300 steps
            self.b0 = 0.2
            self.b1 = 0.005
            self.b2 = 0.05
            self.c1 = 0.01
            self.b3 = 0.03
        elif self.battery_consumption == 'medium' :  # environ 1000 steps
            self.b0 = 0.06
            self.b1 = 0.001
            self.b2 = 0.04
            self.c1 = 0.01
            self.b3 = 0.04
        elif self.base_consumption_rate == 'easy' :  # environ 4000 steps
            self.b0 = 0.01
            self.b1 = 0.0005
            self.b2 = 0.006
            self.c1 = 0.001
            self.b3 = 0.006

        self.max_steps = max_steps # paramètre à définir entre 0 et +inf si on veut terminer la course "brutalement" 
        


        
        
  
        
        
        
    
    
    def reset(self, nb_agents=4, seed=None, nb_steps=1100, render_level=[1, 1, 0]):# cf. https://openprompt.co/conversations/4172
               
        super().reset(seed=seed) #cf. https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        self.nb_agents = nb_agents
        self.render_level = render_level  # array de booléen pour savoir quelles parties afficher # 1 : afficher carte et robot 2 : afficher cone de détection 3 : afficher aiguilles etc
        self.remaining_time = nb_steps # sert à définir un chrono pour la course (inutile sauf dans render)
       
        self.maps_index = (self.maps_index+1)%self.nb_maps 
        
        # on récupère la carte [maps_index]
        maps_infos = self.maps[self.maps_index]
        if maps_infos[5] == 1 or maps_infos[6] == 100 : # SOIT ON A REUSSI LA MAPS SOIT ON A DEJA TENTE 100 FOIS
            # si TRUE : on génère une nouvelle carte qui vient remplacer l'ancienne dans les  "self.nb_maps" qui tournent
            small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos = generate_circuit(seed = self.seeds[self.current_max_seeds], a = self.map_counter)
            self.maps[self.maps_index] = [small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]
            self.current_max_seeds += 1
            maps_infos = self.maps[self.maps_index]
            self.map_counter+=1
        

        # chargement des infos de la carte
        self.small_hexagon = maps_infos[0]
        self.big_hexagon = maps_infos[1]
        self.starting_zone_polygon = maps_infos[2]
        self.bouees = maps_infos[3]
        self.agent_positions = maps_infos[4]
        self.maps[self.maps_index][6] += 1 # on incrémente le nombre de try sur la map
        self.timesteps = 0

        
        self.gateways, self.rayon_gateways = get_gateways(self.big_hexagon, self.small_hexagon)

        # définitions de variables relatives aux agents
        self.current_gateway = np.tile(self.gateways[0], (self.nb_agents, 1))
        self.next_gateways = np.tile(self.gateways[1], (self.nb_agents, 1))
        self.current_gateway_index = np.zeros(self.nb_agents, dtype=np.int8)
        self.agent_speeds = np.random.randint(0, 30, self.nb_agents)
        self.nb_gateway_reached = np.zeros(self.nb_agents)
        self.nb_laps = np.zeros(self.nb_agents)
        self.rewards = np.zeros(self.nb_agents)
        self.sum_rewards = np.zeros(self.nb_agents)
        self.dones = np.zeros(self.nb_agents)
        self.nb_steps = np.zeros(self.nb_agents)
        self.dejadones = np.zeros(self.nb_agents)
        self.battery_level = np.ones(self.nb_agents)*100
        self.rayon_bouee = 23

        # stockage de stats :
        self.nb_colisions = np.zeros(self.nb_agents)
        self.nb_pannes = np.zeros(self.nb_agents)
        

        # calcul matriciel des directions initiales des agents et leurs distances à l'objectif 
        self.directions_objectif = np.arctan2(self.current_gateway[:, 1]-self.agent_positions[:, 1], self.current_gateway[:,0]-self.agent_positions[:, 0])
        self.agent_directions = self.directions_objectif.copy()    
        self.directions_objectif_suivant = np.arctan2(self.next_gateways[:, 1] - self.agent_positions[:, 1], self.next_gateways[:, 0] - self.agent_positions[:, 0])
        # 
        dists = np.sqrt(np.sum((np.expand_dims(self.agent_positions, axis=1) - np.expand_dims(self.current_gateway, axis=0))**2, axis=2))  # result 4, 4, 2
        eye = np.eye(self.nb_agents)  # 4, 4
        diag = dists[np.where(eye)] / np.sqrt(self.screen_width_ned**2 + self.screen_height_ned**2)
        self.current_gateway_distance = diag

    
        return self.get_env_state()
          
    

    def get_env_state(self):
        """
        récupère l'état de l'environnement pour tous les agents d'un coup
        => cela comprend un state1 de l'observation conique des robots (forme d'un vecteur 220,) & une array de variables (forme 6,)
        
        """


        # initialisation du state1 
        state1 = np.zeros((self.nb_agents, 11, 20))
        
        # initialisation du state2 et remplissage avec les informations d'agent
        state2 = np.zeros((self.nb_agents, 6), dtype=np.float32)
        state2[:,0] = self.agent_directions / (2*np.pi)
        state2[:,1] = self.agent_speeds/15
        state2[:,2] = self.battery_level/100
        state2[:,3] = self.directions_objectif / (2*np.pi)
        state2[:,4] = self.directions_objectif_suivant / (2*np.pi)
        state2[:,5] = self.current_gateway_distance 
                   

        # AVEC NUMPY ET CALCUL MATRICIEL : on calcule tous les points dans le cone de vision des agents
        angles = np.linspace(-11*np.pi/36, 11*np.pi/36, 11)   # 11
        distances = np.linspace(2, 40, 20)  # 20
        agents_cones = np.tile(angles, [self.nb_agents, 1]) + np.expand_dims(self.agent_directions, axis=1)   # 10, 11
        agents_cones = np.concatenate([np.expand_dims(np.cos(agents_cones), axis=2), np.expand_dims(np.sin(agents_cones), axis=2)], axis=2)
        all_points = np.expand_dims(np.expand_dims(self.agent_positions, axis=1), axis=1) + np.expand_dims(agents_cones, axis=2) * np.expand_dims(np.tile(distances, [self.nb_agents, 1, 1]), axis=3)


        # avec boucle for on parcourt pour chaque agent si un de ses points est proche d'un obstacle ou d'un concurrent
        for i in range(self.nb_agents) :
            concurrents_positions = np.concatenate([self.agent_positions[0:i], self.agent_positions[i+1:]])

            # Calcul distances     des points du cone à concurrents donc 11, 20, 2 à 9, 2   => 11, 20, 1, 2  et 1, 1, 9, 2  => avec somme 11, 20, 9 + cond et any   11, 20
            my_points = all_points[i]   # 11, 20, 2   
            dist_a_conc = np.any(np.sum( (np.expand_dims(my_points, axis=2) - np.expand_dims(np.expand_dims(concurrents_positions, axis=0), axis=0)   )**2 , axis=3  ) < 10**2, axis=2)
            dist_a_boue = np.any(np.sum( (np.expand_dims(my_points, axis=2) - np.expand_dims(np.expand_dims(self.bouees, axis=0), axis=0)   )**2 , axis=3  ) < 23**2, axis=2)

            # On a 2 mask de booléens : union puis propagation
            double_mask = np.concatenate([np.expand_dims(dist_a_conc, axis=-1), np.expand_dims(dist_a_boue, axis=-1)], axis=-1)
            mask_final = np.any(double_mask, axis=2)

            indices = np.argwhere(mask_final)
            ind_direc = np.unique(indices[:, 0])  # LES INDICES DE DIRECTIONS CONCERNES PAR UN OBSTACLE
            # 2ème boucle sert pour la propagation : si à 3m on a un obstacle on détecte des obstacles sur toute la direction
            for ind in ind_direc :
                first = np.min(np.squeeze(np.argwhere(indices[:, 0]==ind), axis=1))
                dist_min = indices[first]
                state1[i, dist_min[0], dist_min[1]:] = 2
        
        # on récupère tous les points qui ne sont pas bloquées pas un obstacle
        points_restants = np.argwhere(state1==0)
        # et on vérifie si ils sont dans le circuit (0) ou en dehors (1)
        for p in points_restants :
            point = all_points[p[0], p[1], p[2]]
            if self.nb_gateway_reached[i] > 0 :
                # si on a atteint 1 point de passage alors on est dans le circuit et la zone de depart disparait
                if is_inside_hexagon(self.small_hexagon, point) :
                    state1[p[0], p[1], p[2]] = 1
                elif not is_inside_hexagon(self.big_hexagon, point) :
                    state1[p[0], p[1], p[2]] = 1
                    
            else :
                if not is_inside_hexagon(self.starting_zone_polygon, point) :
                    if not is_inside_hexagon(self.big_hexagon, point) or is_inside_hexagon(self.small_hexagon, point)  :
                        state1[p[0], p[1], p[2]] = 1

        state1 = state1.reshape(self.nb_agents, 220) # nb agents, 220  (équivalent à flatten mais conserve dim de l'agent)
        self.state1 = state1[self.followed] # on stocke le cone de l'agent "followed" pour afficher dans le render
        state = [state1, state2]

       
        return state
    
        
    def step(self, action):
        """
        fonction de step qui prend en entrée une matrice d'actions de shape nb_agents, 2 
        avec une vitesse en 0 et un angle en 1 
        """

        self.dejadones = self.dones # on synchronise les dejadones avec les dones pour ne plus jouer les robots dones
        self.rewards = np.zeros((self.nb_agents))

        # on calcules nouvelles vitesses, directions et positions  à partir des actions
        self.agent_speeds = np.clip(self.agent_speeds+action[:, 0], 0, 30)  # entre 0 et 30 km
        self.agent_directions = (self.agent_directions+np.deg2rad(action[:, 1])) % (2*np.pi)
        conso = (self.b0 + self.agent_speeds*self.b1 + self.b2*abs(action[:, 0]+self.c1) + abs(np.deg2rad(action[:, 1]))*self.b3)
        self.battery_level = np.clip(self.battery_level-conso, 0, 100)
        new_coords = np.clip(self.agent_positions + np.concatenate([np.expand_dims(np.cos(self.agent_directions)*self.agent_speeds/3.6, axis=1), np.expand_dims(np.sin(self.agent_directions)*self.agent_speeds/3.6, axis=1)], axis=1), [0, -self.screen_height_ned], [self.screen_width_ned, 0])
        # division par 3.6 on consière qu'on fait 1 step par seconde 
        dist_bouees = np.sum((np.expand_dims(new_coords, axis=1) - np.expand_dims(self.bouees, axis=0))**2, axis=2)
        dist_agents = np.sum((np.expand_dims(new_coords, axis=1) - np.expand_dims(new_coords, axis=0))**2, axis=2)


        followed_attributed = False  # on cherche à attribuer followed à un agent vivant
        for i in range(self.nb_agents) :
            reward = 0
            if not self.dejadones[i] :  # si l'agent n'est pas déjà mort au tour d'avant alors on calcule son reward
                self.nb_steps[i] += 1
                if not followed_attributed :
                    self.followed = i
                    followed_attributed = True
                self.agent_positions[i] = new_coords[i]  # on attribue les coordonnées

                if np.any(dist_bouees[i] < 529) :   # si colision avec bouee
                    if self.timesteps > 20 : # si moins de 20 steps les colisions ne tuent pas
                      self.nb_colisions+=1
                      self.dones[i] = True
                    reward -= 12
                else :  
                    if np.any(np.concatenate([dist_agents[i, i+1:], dist_agents[i, :i]])<100) :  # si colision avec concurrent
                        if self.timesteps>20 :
                            self.nb_colisions+=1
                            self.dones[i] = True
                        reward -= 8
                    else :  # si aucune colision
                        
                        if np.sum((self.agent_positions[i] - self.current_gateway[i])**2) < self.rayon_gateways[self.current_gateway_index[i]]**2 :
                            reward+= 10 
                            self.nb_gateway_reached[i] +=1
                            self.current_gateway_index[i] = ((self.current_gateway_index[i]+1)%len(self.gateways))
                            self.current_gateway[i] = self.gateways[self.current_gateway_index[i]]
                            self.next_gateways[i] = self.gateways[(self.current_gateway_index[i]+1)%len(self.gateways)]
                            self.nb_laps[i] = (self.nb_gateway_reached[i]-1)//6
                            if (self.nb_gateway_reached[i]-1) % 6 == 0 and self.nb_gateway_reached[i] > 1 :
                                print("l'agent a fait un tour!!!")
                                self.maps[self.maps_index][5] = 1  # on considère la carte comme achevé car un agent a réussi à faire un tour
                        

                        # VERIFICATION BATTERIE
                        if self.battery_level[i] < 1 :  # mort si moins de 1%
                            reward -= 3
                            self.dones[i] = True
                            self.nb_pannes += 1

                        # INTEGRATION DISTANCE DANS REWARD AVEC PALIER SUR LA DISTANCE NORMALISEE
                        gateway_distance = np.sum(np.abs(np.array(self.current_gateway[i]) - np.array(self.agent_positions[i])))  / np.sqrt(self.screen_width_ned**2 + self.screen_height_ned**2)
                        if gateway_distance < 0.5 :
                            reward -= 0.6
                        else :
                            reward -= 1.2
                        
                        # ATTRIBUE UN REWARD SELON LA POSITION (dans ou hors circuit)
                        if self.nb_gateway_reached[i] == 0 :
                            if is_inside_hexagon(self.starting_zone_polygon, self.agent_positions[i]) :
                                reward -= 0.5
                            else :
                                reward -= 2
                        else :
                            if is_inside_hexagon(self.big_hexagon, self.agent_positions[i]) :
                                if is_inside_hexagon(self.small_hexagon, self.agent_positions[i]) :
                                    reward -= 2
                                else :
                                    reward -= 0.5
                            else :
                                reward -= 2
                
        
                    
                self.rewards[i] = reward/40   # on divise les reward par 40 pour qu'ils soient proches de 0 et on le stocke dans l'attribut d'environnement
               

        self.sum_rewards += self.rewards  # stocke la somme des reward pour un épisode 
        # recalcul attributs de directions et distances en matriciel
        self.directions_objectif = np.arctan2(self.current_gateway[:, 1]-self.agent_positions[:, 1], self.current_gateway[:,0]-self.agent_positions[:, 0])
        self.directions_objectif_suivant = np.arctan2(self.next_gateways[:, 1] - self.agent_positions[:, 1], self.next_gateways[:, 0] - self.agent_positions[:, 0])
        self.timesteps+=1

        # 4, 1, 2 et 1, 4, 2 ==> 4, 4
        dists = np.sqrt(np.sum((np.expand_dims(self.agent_positions, axis=1) - np.expand_dims(self.current_gateway, axis=0))**2, axis=2))  # result 4, 4, 2
        # les distance de l'agent i à chaque gateway, ce qui nous intéresse c'est les distances de l'agent i à la gateway i donc la diagonale
        eye = np.eye(self.nb_agents)  # 4, 4
        diag = dists[np.where(eye)] / np.sqrt(self.screen_width_ned**2 + self.screen_height_ned**2) 
        self.current_gateway_distance = diag

        return self.get_env_state(), self.rewards, self.dejadones, self.dones, "", ""
       

    def gen_new_map(self, seed) :
        # permet de générer une nouvelle carte pour écraser l'actuelle 0, fonction utilisée principalement pour les inférences
        if seed == None :
            self.maps[0] = [self.real_small_h, self.real_big_h, self.real_start_zone, self.real_bouees, self.real_init_pos, 0, 0]
        else: 
            small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos =  generate_circuit(seed)
            self.maps[0] = [small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]
        self.maps_index = -1

    def get_metriques(self):
        return self.nb_gateway_reached, self.nb_laps, self.nb_steps, self.sum_rewards
    

    def inference(self) :
        self.small_hexagon = self.real_small_h
        self.big_hexagon = self.real_big_h
        self.starting_zone_polygon = self.starting_zone_polygon
        self.bouees = self.real_bouees
        self.agent_positions = generate_points_in_quadrilateral(self.starting_zone_polygon, self.nb_agents, 30)

        self.gateways, self.rayon_gateways = get_gateways(self.big_hexagon, self.small_hexagon)
        self.current_gateway = np.tile(self.gateways[0], (self.nb_agents, 1))
        self.next_gateways = np.tile(self.gateways[1], (self.nb_agents, 1))
        self.agent_speeds = np.zeros(self.nb_agents)        

        self.directions_objectif = np.arctan2(self.current_gateway[:, 1]-self.agent_positions[:, 1], self.current_gateway[:,0]-self.agent_positions[:, 0])
        self.agent_directions = self.directions_objectif.copy()    
        self.directions_objectif_suivant = np.arctan2(self.next_gateways[:, 1] - self.agent_positions[:, 1], self.next_gateways[:, 0] - self.agent_positions[:, 0])
        dists = np.sqrt(np.sum((np.expand_dims(self.agent_positions, axis=1) - np.expand_dims(self.current_gateway, axis=0))**2, axis=2))
        eye = np.eye(self.nb_agents) 
        diag = dists[np.where(eye)] / np.sqrt(self.screen_width_ned**2 + self.screen_height_ned**2)
        self.current_gateway_distance = diag
        

    def render(self):
        """
        fonction render qui produit retour graphique sur l'environnement
        utilise variable "render_level" pour décider quelles parties dessiner (très couteux tout à chaque fois + dessins sur les circuits aléatoires donc masqués)
        => fait render en utilisant PIL 
        pour utiliser la fonction faire env.render() puis ensuite récupérer l'attribut env.frame dans une liste (en stockant N frames produit gif)
        """
        if self.render_flag:
            # Dimensions de l'écran
            if self.render_level[0] :

                width, height = self.screen_width_ned, self.screen_height_ned
                
                # Charger l'image de fond
                bg = Image.open(self.monaco_map).convert("RGBA")
                draw = ImageDraw.Draw(bg)
                
                draw_bouees(draw, ned_to_image(self.bouees[:, 0], self.bouees[:, 1], self.pixels_per_meter), self.rayon_bouee)
                draw_current_gateway(draw, colors, ned_to_image(self.current_gateway[:, 0], self.current_gateway[:, 1], self.pixels_per_meter), self.rayon_gateways[self.current_gateway_index])
                draw_circuit(draw, ned_to_image(self.big_hexagon[:, 0], self.big_hexagon[:, 1], self.pixels_per_meter), 
                             ned_to_image(self.small_hexagon[:, 0], self.small_hexagon[:, 1], self.pixels_per_meter),
                              ned_to_image(self.starting_zone_polygon[:, 0], self.starting_zone_polygon[:, 1], self.pixels_per_meter))
                
                draw_boat_circle(draw, ned_to_image(self.agent_positions[:, 0], self.agent_positions[:, 1], self.pixels_per_meter), self.dones, colors)
                draw_boat_triangle(draw, ned_to_image(self.agent_positions[:, 0], self.agent_positions[:, 1], self.pixels_per_meter), self.agent_directions, self.dones, colors)
                draw_detection_field(draw, ned_to_image(self.agent_positions[:, 0], self.agent_positions[:, 1], self.pixels_per_meter), self.agent_directions, self.dones, colors)

                if self.render_level[1] :
                    draw_detections_table(draw, self.font, np.reshape(self.state1, [220]))
                if self.render_level[2] :
                    draw_agent_info(draw, self.font, self.agent_directions[self.followed], self.agent_speeds[self.followed], self.remaining_time, self.nb_laps, self.nb_gateway_reached, self.battery_level,
                                    self.rewards, self.directions_objectif[self.followed], self.agent_positions[self.followed, 0], self.agent_positions[self.followed, 1], colors[self.followed])
                    
            
                self.frame = bg



