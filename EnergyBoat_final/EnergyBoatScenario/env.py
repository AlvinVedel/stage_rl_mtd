from copy import deepcopy
import yaml
import time
import sys
import math
import random
import time
import numpy as np
from pygame.locals import QUIT
import gymnasium as gym
from PIL import Image, ImageDraw


from EnergyBoatScenario.bridge import *
from EnergyBoatScenario.utils_env import *
from EnergyBoatScenario.view2d import *


'''
In this file, you must write the 4 functions: init, reset, step and render.
Other functions may be written in the corresponding utils_env.py file
'''

#import tensorflow as tf
#from tensorflow.keras import layers

actionSet = {
    1: [-1.5, -15],
    2: [-1.5, -6],
    3: [-1.5, 0],
    4: [-1.5, 6],
    5: [-1.5, 15],
    6: [-0.6, -15],
    7: [-0.6, -6],
    8: [-0.6, 0],
    9: [-0.6, 6],
    10: [-0.6, 15],
    11: [0, -15],
    12: [0, -6],
    13: [0, 0],
    14: [0, 6],
    15: [0, 15],
    16: [0.6, -15],
    17: [0.6, -6],
    18: [0.6, 0],
    19: [0.6, 6],
    20: [0.6, 15],
    21: [1.5, -15],
    22: [1.5, -6],
    23: [1.5, 0],
    24: [1.5, 6],
    0: [1.5, 15]
}
colors = ['red', 'blue', 'green', 'yellow', 'white', 'black', 'purple', 'orange']



def area_triangle(point1, point2, point3):
    return 0.5 * np.abs((point1[0] - point3[0]) * (point2[1] - point1[1]) - (point1[0] - point2[0]) * (point3[1] - point1[1]))

def area_quadrilataire(p1, p2, p3, p4):
    a1 = area_triangle(p1, p2, p3)
    a2 = area_triangle(p3, p4, p1)
    return a1+a2
"""
def is_inside_hexagon(points, test_point):
    total_area = 0
    for i in range(len(points)):
        total_area += area_triangle(points[i], points[(i+1)%len(points)], test_point)
    if len(points) == 4 :
        area = area_quadrilataire(points[0], points[1], points[2], points[3])
    else :
        a1 = area_quadrilataire(points[0], points[1], points[2], points[3])
        a2 = area_quadrilataire(points[3], points[4], points[5], points[0])
        area = a1+a2
    return np.abs(total_area - area) < 0.01
"""
def is_inside_hexagon(vertices, point):
    """
    Détermine si un point est à l'intérieur d'un polygone non convexe.
    
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


def random_point_in_triangle(triangle):
    r1 = random.random()
    r2 = random.random()
    if r1 + r2 >= 1:
        r1 = 1 - r1
        r2 = 1 - r2
    a, b, c = triangle
    x = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
    y = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
    return (x, y)

def random_point_in_quadrilateral(quadrilateral):
    tri1 = [quadrilateral[0], quadrilateral[1], quadrilateral[2]]
    tri2 = [quadrilateral[0], quadrilateral[2], quadrilateral[3]]
    if random.random() < 0.5:
        return random_point_in_triangle(tri1)
    else:
        return random_point_in_triangle(tri2)
    
"""
def random_init_position(nb_agents, quadrilateral) :
    points = quadrilateral

    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)

    x = np.arange(x_min, x_max, 25)
    y = np.arange(y_min, y_max, 25)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    selected = np.random.choice(np.arange(grid_points.shape[0]), replace=False, size=(nb_agents))
    points_selected = grid_points[selected]
    

    return points_selected
"""
def generate_random_point_in_quadrilateral(p1, p2, p3, p4):
    """Generate a random point inside a quadrilateral using bilinear interpolation."""
    r1 = random.random()
    r2 = random.random()
    
    x = (1 - r1) * (1 - r2) * p1[0] + r1 * (1 - r2) * p2[0] + r1 * r2 * p3[0] + (1 - r1) * r2 * p4[0]
    y = (1 - r1) * (1 - r2) * p1[1] + r1 * (1 - r2) * p2[1] + r1 * r2 * p3[1] + (1 - r1) * r2 * p4[1]
    
    return np.array([x, y])

def generate_points_in_quadrilateral(sz, nb_points, dist):
      diag = np.eye(nb_points)*dist**3

      while True : 
        points = np.zeros((nb_points, 2))  # 4, 2  => 1, 4, 2
        for i in range(nb_points) :
          points[i] = generate_random_point_in_quadrilateral(sz[0], sz[1], sz[2], sz[3])
        distance = np.sum((np.expand_dims(points, axis=0) - np.expand_dims(points, axis=1))**2, axis=2)   # 1, 4, 2  et 4, 1, 2  ==> 4, 4, 2 puis somme axe 2 => 4, 4
        nd = distance + diag 
        if np.all(nd >= dist**2):
          return points


def generate_circuit(seed=42, a=0, nb_agents=4, distance_agents=30) :
    random.seed(seed)
    np.random.seed(seed+10)
    a = math.sqrt(2*a)
    max_x = 1280
    max_y = 720
    points = []
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
    orthogonal = np.array([-vector[1], vector[0]])
    unit_vector = orthogonal / np.linalg.norm(orthogonal)
    scaled_vector = unit_vector * 300
    sz1 = np.array(big_hexagon[0]) + scaled_vector
    sz2 = np.array(small_hexagon[0]) + scaled_vector

    start_zone = np.array([sz1, big_hexagon[0], small_hexagon[0], sz2])


    init_pos = generate_points_in_quadrilateral(start_zone, nb_agents, distance_agents)
    return small_hexagon, big_hexagon, start_zone, bouees, init_pos



def tuple_polygon_to_array(polygon):
    """
    fonction qui convertis des polygones tuples en matrices numpy (pour charger les données du yaml)
    """
    array_polygon = np.zeros((len(polygon), 2))
    for i in range(len(polygon)) :
        array_polygon[i] = np.array([polygon[i][0], polygon[i][1]])
    return array_polygon


class EnergyBoatEnv(gym.Env) : 

    def __init__(self, env_config, nb_agents=4, recharge=False, battery_consumption='medium', max_steps=1100, know_time=False, init_seed=42) :
        
        # CHARGEMENT DE CERTAINS PARAMETRES DU YAML
        self.font = env_config["font"]
        self.monaco_map = env_config["monaco_map"]
        self.distance_ref_px = env_config["distance_ref_px"] 
        self.distance_ref_m = env_config["distance_ref_m"]
        self.pixels_per_meter = self.distance_ref_px / self.distance_ref_m

        # RECUPERATION ET STOCKAGE DU VRAI CIRCUIT DE LA COURSE
        self.real_big_h = tuple_polygon_to_array(env_config["big_hexagon"])
        self.real_small_h = tuple_polygon_to_array(env_config["small_hexagon"])

        self.real_start_zone = tuple_polygon_to_array(env_config["starting_zone_polygon"])
        self.real_init_pos = generate_points_in_quadrilateral(self.real_start_zone, nb_agents, 30)

        self.real_rayon_bouee = env_config["rayon_bouee"]
        self.real_bouees = tuple_polygon_to_array(env_config["bouees"])
        
        self.screen_width_ned = SCREEN_WIDTH/self.pixels_per_meter  # IMPORTANT
        self.screen_height_ned = SCREEN_HEIGHT/self.pixels_per_meter   # SIGNE - car NED inverse les coordonnées y dans le clip
        
        self.render_flag= True

        ########### ATTRIBUTS AJOUTES #########
        self.nb_agents = nb_agents
        self.recharge = recharge
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

        for i in range(self.nb_maps) :
            small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos = generate_circuit(seed = self.seeds[i])
            self.maps.append([small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]) # LE PREMIER 0 EST UN BOOLEEN QUI DIT SI LA MAP A ETE COMPLETE, LE 2EME POUR LE NOMBRE DE TRY SUR LA MAP
            self.current_max_seeds += 1 # = 10
        #print("SEEDS",self.seeds[:100])
       
        self.followed = 0
        #self.small_hexagon, self.big_hexagon, self.starting_zone_polygon, self.bouees = generate_circuit()

        
          
        self.battery_consumption = battery_consumption
        if self.battery_consumption == 'hard' :
            self.b0 = 0.2
            self.b1 = 0.005
            self.b2 = 0.05
            self.c1 = 0.01
            self.b3 = 0.03
        elif self.battery_consumption == 'medium' :
            self.b0 = 0.06
            self.b1 = 0.001
            self.b2 = 0.04
            self.c1 = 0.01
            self.b3 = 0.04
        elif self.base_consumption_rate == 'easy' :
            self.b0 = 0.01
            self.b1 = 0.0005
            self.b2 = 0.006
            self.c1 = 0.001
            self.b3 = 0.006

        self.max_steps = max_steps
        self.pixels_per_meter = 1
        


        
        
  
        
        
        
    
    
    def reset(self, nb_agents=4, seed=None, return_info=False, options=None, rendering=False, nb_steps=1100, render_level=[1, 0, 0]):# cf. https://openprompt.co/conversations/4172
               
        super().reset(seed=seed) #cf. https://gymnasium.farama.org/api/env/#gymnasium.Env.reset
        self.nb_agents = nb_agents
        self.rendering = rendering
        self.render_level = render_level
        self.remaining_time = nb_steps
       
        self.maps_index = (self.maps_index+1)%self.nb_maps 
        
        maps_infos = self.maps[self.maps_index]
        if maps_infos[5] == 1 or maps_infos[6] == 100 : # SOIT ON A REUSSI LA MAPS SOIT ON A DEJA TENTE 100 FOIS
            small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos = generate_circuit(seed = self.seeds[self.current_max_seeds], a = self.map_counter)
            self.maps[self.maps_index] = [small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]
            self.current_max_seeds += 1
            maps_infos = self.maps[self.maps_index]
            self.map_counter+=1
        
    
        self.small_hexagon = maps_infos[0]
        self.big_hexagon = maps_infos[1]
        self.starting_zone_polygon = maps_infos[2]
        self.bouees = maps_infos[3]
        self.agent_positions = maps_infos[4]
        self.maps[self.maps_index][6] += 1
        self.timesteps = 0

        
        self.gateways, self.rayon_gateways = get_gateways(self.big_hexagon, self.small_hexagon)


        self.current_gateway = np.tile(self.gateways[0], (self.nb_agents, 1))
        self.next_gateways = np.tile(self.gateways[1], (self.nb_agents, 1))
        self.current_gateway_index = np.zeros(self.nb_agents, dtype=np.int8)
        self.agent_speeds = np.zeros(self.nb_agents)
        self.nb_gateway_reached = np.zeros(self.nb_agents)
        self.nb_laps = np.zeros(self.nb_agents)
        self.rewards = np.zeros(self.nb_agents)
        self.sum_rewards = np.zeros(self.nb_agents)
        self.dones = np.zeros(self.nb_agents)
        self.nb_steps = np.zeros(self.nb_agents)
        self.dejadones = np.zeros(self.nb_agents)
        self.battery_level = np.ones(self.nb_agents)*100
        self.rayon_bouee = 23

        self.nb_colisions = np.zeros(self.nb_agents)
        self.nb_pannes = np.zeros(self.nb_agents)
              
        self.directions_objectif = np.arctan2(self.current_gateway[:, 1]-self.agent_positions[:, 1], self.current_gateway[:,0]-self.agent_positions[:, 0])
        self.agent_directions = self.directions_objectif.copy()    
        self.directions_objectif_suivant = np.arctan2(self.next_gateways[:, 1] - self.agent_positions[:, 1], self.next_gateways[:, 0] - self.agent_positions[:, 0])
        dists = np.sqrt(np.sum((np.expand_dims(self.agent_positions, axis=1) - np.expand_dims(self.current_gateway, axis=0))**2, axis=2))  # result 4, 4, 2
        eye = np.eye(self.nb_agents)  # 4, 4
        diag = dists[np.where(eye)] / np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2)
        self.current_gateway_distance = diag
        self.reward_factor = 1
    
        return self.get_env_state()
          
    

    def get_env_state(self):

        state1 = np.zeros((self.nb_agents, 11, 20))
       
        state2 = np.zeros((self.nb_agents, 6), dtype=np.float32)
        state2[:,0] = self.agent_directions / (2*np.pi)
        state2[:,1] = self.agent_speeds/15
        state2[:,2] = self.battery_level/100
        state2[:,3] = self.directions_objectif / (2*np.pi)
        state2[:,4] = self.directions_objectif_suivant / (2*np.pi)
        state2[:,5] = self.current_gateway_distance 
                   

        angles = np.linspace(-11*np.pi/36, 11*np.pi/36, 11)  
        distances = np.linspace(2, 40, 20)  
    
        agents_cones = np.tile(angles, [self.nb_agents, 1]) + np.expand_dims(self.agent_directions, axis=1)   # 10, 11
        agents_cones = np.concatenate([np.expand_dims(np.cos(agents_cones), axis=2), np.expand_dims(np.sin(agents_cones), axis=2)], axis=2)
        all_points = np.expand_dims(np.expand_dims(self.agent_positions, axis=1), axis=1) + np.expand_dims(agents_cones, axis=2) * np.expand_dims(np.tile(distances, [self.nb_agents, 1, 1]), axis=3)


        for i in range(self.nb_agents) :
            concurrents_positions = np.concatenate([self.agent_positions[0:i], self.agent_positions[i+1:]])

            my_points = all_points[i]   
            dist_a_conc = np.any(np.sum( (np.expand_dims(my_points, axis=2) - np.expand_dims(np.expand_dims(concurrents_positions, axis=0), axis=0)   )**2 , axis=3  ) < 10**2, axis=2)
            dist_a_boue = np.any(np.sum( (np.expand_dims(my_points, axis=2) - np.expand_dims(np.expand_dims(self.bouees, axis=0), axis=0)   )**2 , axis=3  ) < 23**2, axis=2)

            double_mask = np.concatenate([np.expand_dims(dist_a_conc, axis=-1), np.expand_dims(dist_a_boue, axis=-1)], axis=-1)
            mask_final = np.any(double_mask, axis=2)

            indices = np.argwhere(mask_final)
            ind_direc = np.unique(indices[:, 0])  
            for ind in ind_direc :
                first = np.min(np.squeeze(np.argwhere(indices[:, 0]==ind), axis=1))
                dist_min = indices[first]
                state1[i, dist_min[0], dist_min[1]:] = 2
                
        points_restants = np.argwhere(state1==0)
        for p in points_restants :
            point = all_points[p[0], p[1], p[2]]
            if self.nb_gateway_reached[i] > 0 :
                if is_inside_hexagon(self.small_hexagon, point) :
                    state1[p[0], p[1], p[2]] = 1
                elif not is_inside_hexagon(self.big_hexagon, point) :
                    state1[p[0], p[1], p[2]] = 1

            else :
                if not is_inside_hexagon(self.starting_zone_polygon, point) :
                    if not is_inside_hexagon(self.big_hexagon, point) or is_inside_hexagon(self.small_hexagon, point)  :
                        state1[p[0], p[1], p[2]] = 1
        state1 = state1.reshape(self.nb_agents, 220) 
        self.state1 = state1[self.followed]
        state = [state1, state2]


        return state
    
        
    def step(self, action):
        
        self.dejadones = self.dones
        self.rewards = np.zeros((self.nb_agents))

        self.agent_speeds = np.clip(self.agent_speeds+action[:, 0], 0, 15)
        self.agent_directions = (self.agent_directions+np.deg2rad(action[:, 1])) % (2*np.pi)
        conso = (self.b0 + self.agent_speeds*self.b1 + self.b2*abs(action[:, 0]+self.c1) + abs(np.deg2rad(action[:, 1]))*self.b3)
        self.battery_level = np.clip(self.battery_level-conso, 0, 100)
        new_coords = np.clip(self.agent_positions + np.concatenate([np.expand_dims(np.cos(self.agent_directions)*self.agent_speeds, axis=1), np.expand_dims(np.sin(self.agent_directions)*self.agent_speeds, axis=1)], axis=1), [0, 0], [SCREEN_WIDTH, SCREEN_HEIGHT])
        dist_bouees = np.sum((np.expand_dims(new_coords, axis=1) - np.expand_dims(self.bouees, axis=0))**2, axis=2)
        dist_agents = np.sum((np.expand_dims(new_coords, axis=1) - np.expand_dims(new_coords, axis=0))**2, axis=2)

        followed_attributed = False

        for i in range(self.nb_agents) :
            reward = 0
            if not self.dejadones[i] :
                self.nb_steps[i] +=1
                if not followed_attributed :
                    self.followed = i
                    followed_attributed = True
                self.agent_positions[i] = new_coords[i]
                if np.any(dist_bouees[i] < 529) :  
                    if self.timesteps > 20 :
                      self.nb_colisions[i]+=1
                      self.dones[i] = True
                    reward -= 10
                else :
                    if np.any(np.concatenate([dist_agents[i, i+1:], dist_agents[i, :i]])<100) :
                        if self.timesteps>20 :
                            self.nb_colisions[i]+=1
                            self.dones[i] = True
                        reward -= 10
                    else :
                        
                        if np.sum((self.agent_positions[i] - self.current_gateway[i])**2) < self.rayon_gateways[self.current_gateway_index[i]]**2 :
                            reward+= 10 
                            self.nb_gateway_reached[i] +=1
                            self.current_gateway_index[i] = ((self.current_gateway_index[i]+1)%len(self.gateways))
                            self.current_gateway[i] = self.gateways[self.current_gateway_index[i]]
                            self.next_gateways[i] = self.gateways[(self.current_gateway_index[i]+1)%len(self.gateways)]
                            self.nb_laps[i] = (self.nb_gateway_reached[i]-1)//6
                            if (self.nb_gateway_reached[i]-1) % 6 == 0 and self.nb_gateway_reached[i] > 1 :
                                print("l'agent a fait un tour!!!")
                               
                                self.maps[self.maps_index][5] = 1
                        
                        
                        # VERIFICATION BATTERIE
                        if self.battery_level[i] < 1 :
                            reward -= 5
                            self.dones[i] = True
                            self.nb_pannes[i] += 1

                       

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
                
        
                    
                self.rewards[i] = reward*self.reward_factor/40
                
        

        self.sum_rewards += self.rewards
        self.directions_objectif = np.arctan2(self.current_gateway[:, 1]-self.agent_positions[:, 1], self.current_gateway[:,0]-self.agent_positions[:, 0])
        self.directions_objectif_suivant = np.arctan2(self.next_gateways[:, 1] - self.agent_positions[:, 1], self.next_gateways[:, 0] - self.agent_positions[:, 0])
        self.timesteps+=1

        
        dists = np.sqrt(np.sum((np.expand_dims(self.agent_positions, axis=1) - np.expand_dims(self.current_gateway, axis=0))**2, axis=2)) 
        eye = np.eye(self.nb_agents)  
        diag = dists[np.where(eye)] / np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2) 
        self.current_gateway_distance = diag

        return self.get_env_state(), self.rewards, self.dejadones, self.dones, "", ""
       

    def gen_new_map(self, seed) :
        small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos =  generate_circuit(seed)
        self.maps[0] = [small_hexagon, big_hexagon, starting_zone_polygon, bouees, init_pos, 0, 0]
        self.maps_index = -1

    def get_metriques(self):
        return self.nb_gateway_reached, self.nb_laps, self.nb_steps, self.sum_rewards
    
    def get_inference_metriques(self):
        return self.nb_gateway_reached, self.nb_steps, self.nb_pannes, self.nb_colisions


    def inference(self) :
        self.small_hexagon = self.real_small_h
        self.big_hexagon = self.real_big_h
        self.starting_zone_polygon = self.real_start_zone
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
        if self.render_flag:
            # Dimensions de l'écran
            if self.render_level[0] :

                width, height = SCREEN_WIDTH, SCREEN_HEIGHT
                
                # Charger l'image de fond
                bg = Image.open(self.monaco_map).convert("RGBA")
                draw = ImageDraw.Draw(bg)
                
                draw_bouees(draw, self.bouees, self.rayon_bouee)
                draw_current_gateway(draw, colors, self.current_gateway, self.rayon_gateways[self.current_gateway_index])
                draw_circuit(draw, self.big_hexagon, self.small_hexagon, self.starting_zone_polygon)
                
                draw_boat_circle(draw, self.agent_positions, self.dones, colors)
                draw_boat_triangle(draw, self.agent_positions, self.agent_directions, self.dones, colors)
                draw_detection_field(draw, self.agent_positions, self.agent_directions, self.dones, colors)

                if self.render_level[1] :
                    draw_detections_table(draw, self.font, np.reshape(self.state1, [220]))
                if self.render_level[2] :
                    draw_agent_info(draw, self.font, self.agent_directions[self.followed], self.agent_speeds[self.followed], self.remaining_time, self.nb_laps, self.nb_gateway_reached, self.battery_level,
                                    self.rewards, self.directions_objectif[self.followed], self.agent_positions[self.followed, 0], self.agent_positions[self.followed, 1], colors[self.followed])
                    
            
                self.frame = bg

    
