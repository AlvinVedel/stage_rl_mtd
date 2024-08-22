from PIL import Image
from PIL import ImageDraw
import numpy as np
from MonRobot2 import MonRobot
import tensorflow as tf


def draw_basic_map(size, classic) :
    largeur = 40*size
    hauteur = 40*size
    image = Image.new("RGB", (largeur, hauteur), color=(255, 0, 0))
    draw = ImageDraw.Draw(image)

    points_controle1 = [(12*size, 40*size), (8*size, 33*size), (7*size, 25*size), (14*size, 20*size), (18*size, 10*size), (27*size, 0) ]
    draw.line(points_controle1, fill=(0, 255, 0), width=2)

    points_controle2 = [(21*size, 40*size), (12*size, 33*size), (11*size, 25*size), (19*size, 20*size), (25*size, 10*size), (33*size, 0) ]
    draw.line(points_controle2, fill=(0, 255, 0), width=2)

    all_points = points_controle1 + points_controle2[::-1]  # Inverser la deuxième liste de points
    draw.polygon(all_points, fill=(0, 255, 0))
    if classic :
        cercle_centre = (28*size, 3*size)
        
    else:
        random_point = np.random.randint(largeur, size=2, dtype=np.int16)
        cercle_centre = (random_point[0], random_point[1])
         
    rayon = 1
    draw.ellipse((cercle_centre[0] - rayon, cercle_centre[1] - rayon,
               cercle_centre[0] + rayon, cercle_centre[1] + rayon), fill=(255, 255, 255))
    cercle_point = np.array([cercle_centre[1], cercle_centre[0]], dtype=np.int16)
    return image, cercle_point




class Environnement:
    def __init__(self, size, classic=True):
        self.N = 40*size
        self.size=size
        self.training = classic
        self.agent = MonRobot(position = np.array([39*size, 15*size], dtype=np.float16), size=size)
        self.visuel, self.objectif = draw_basic_map(size=size, classic=classic)  # On stocke le template créé au dessus ici

        self.previous_distance = None 
        self.map = np.zeros((self.N, self.N), dtype=np.int8)
        tableau = np.array(self.visuel)
        self.map[np.all(tableau == [255, 255, 255], axis=-1)] = 2 # Objectif
        self.map[np.all(tableau == [0, 255, 0], axis=-1)] = 1 # Chemin
        self.map[np.all(tableau == [255,0,  0], axis=-1)] = 0 # mur

       
 


    def __str__(self):
        return "L'environnement comporte un agent : "+str(self.agent)
        
    
    

    def reset_env(self):
        self.agent.replace_agent(np.array([39*self.size, 16*self.size], dtype=np.float32))
        if not self.training :
            self.visuel, self.objectif = draw_basic_map(self.size, self.training)
            self.map = np.zeros((self.N, self.N), dtype=np.int8)
            tableau = np.array(self.visuel)
            self.map[np.all(tableau == [255, 255, 255], axis=-1)] = 2 # Objectif
            self.map[np.all(tableau == [0, 255, 0], axis=-1)] = 1 # Chemin
            self.map[np.all(tableau == [255,0,  0], axis=-1)] = 0 # mur

       
    
    def step(self, action):
        
        previous_state = self.map
        previous_agent = [self.agent.position[0], self.agent.position[1], self.agent.vitesse, self.agent.orientation]

        robot_point = self.agent.position
        agent_info = self.map[int(robot_point[0]), int(robot_point[1])]

        if action == 0 :   # GAUCHE ET RALENTIR
            self.agent.rotate_robot(-1)
            self.agent.changer_vitesse(-1)

        elif action == 1 : # GAUCHE ET VITESSE NEUTRE 
            self.agent.rotate_robot(-1)
            self.agent.changer_vitesse(0)

        elif action == 2 :  # GAUCHE ET ACCELERER
            self.agent.rotate_robot(-1)
            self.agent.changer_vitesse(1)

        elif action == 3 :  # ORIENTATION NEUTRE ET RALENTIR
            self.agent.rotate_robot(0)
            self.agent.changer_vitesse(-1)

        elif action == 4 :  # ORIENTATION NEUTRE ET VITESSE NEUTRE
            self.agent.rotate_robot(0)
            self.agent.changer_vitesse(0)

        elif action == 5 :  # ORIENTATION NEUTRE ET ACCELERER
            self.agent.rotate_robot(0)
            self.agent.changer_vitesse(1)

        elif action == 6 :  # DROITE ET RALENTIR
            self.agent.rotate_robot(1)
            self.agent.changer_vitesse(-1)

        elif action == 7 :  # DROITE ET VITESSE NEUTRE
            self.agent.rotate_robot(1)
            self.agent.changer_vitesse(0)

        elif action == 8 :  # DROITE ET ACCELERER
            self.agent.rotate_robot(1)
            self.agent.changer_vitesse(1)

        
        self.agent.avancer(agent_info)
        self.agent.position = np.clip(self.agent.position, 0, int(40*self.size)-1)
        
        distance = 2*np.sum(((self.objectif - self.agent.position)/(40*self.size))**2,  axis=0)        
        reward = 1-distance
        self.previous_distance = distance

        next_agent = np.array([self.agent.position[0], self.agent.position[1], self.agent.vitesse, self.agent.orientation])

        new_state = self.map
        if self.map[int(self.agent.position[0]), int(self.agent.position[1])]!=2 :
            done=0
        else :
            done=1
            reward +=100

        episode = [previous_state, action, reward, new_state, done, previous_agent, next_agent]
        self.agent.remember(episode)
        
        return reward, done