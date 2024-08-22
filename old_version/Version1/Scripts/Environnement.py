from PIL import Image
from PIL import ImageDraw
import numpy as np
from MonRobot import MonRobot
import tensorflow as tf


def draw_basic_map() :
    #print("in draw")
    largeur = 400
    hauteur = 400
    image = Image.new("RGB", (largeur, hauteur), color=(255, 0, 0))
    draw = ImageDraw.Draw(image)

    points_controle1 = [(120, 400), (80, 330), (70, 250), (140, 200), (180, 100), (270, 0) ]
    draw.line(points_controle1, fill=(0, 255, 0), width=2)

    points_controle2 = [(210, 400), (120, 330), (110, 250), (190, 200), (250, 100), (330, 0) ]
    draw.line(points_controle2, fill=(0, 255, 0), width=2)

    all_points = points_controle1 + points_controle2[::-1]  # Inverser la deuxième liste de points
    draw.polygon(all_points, fill=(0, 255, 0))
    cercle_centre = (280, 30)
    rayon = 1
    draw.ellipse((cercle_centre[0] - rayon, cercle_centre[1] - rayon,
               cercle_centre[0] + rayon, cercle_centre[1] + rayon), fill=(255, 255, 255))
    #draw.point(cercle, fill=(255, 255, 255))
    return image




class Environnement:
    def __init__(self):
        self.objectif = np.array([30, 280], dtype=np.int16)
        #self.base =  
        self.N = 400
        self.visuel = draw_basic_map()  # On stocke le template créé au dessus ici
        self.agent = MonRobot(position = np.array([390, 150], dtype=np.int16))
        self.previous_distance = None 
        #self.draw_basic_map()

        self.map = np.zeros((self.N, self.N), dtype=np.int8)
        tableau = np.array(self.visuel)
        self.map[np.all(tableau == [255, 255, 255], axis=-1)] = 2
        self.map[np.all(tableau == [0, 255, 0], axis=-1)] = 1 # Chemin
        self.map[np.all(tableau == [255,0,  0], axis=-1)] = 0 # mur

        #self.map[self.agent.position[0]][self.agent.position[1]]=3

        #position = np.arry()
        #visuelopti = np.array(self.visuel)
        #colors = {
        #        (255, 0, 0): 0,   # Rouge: Mur
        #        (0, 255, 0): 1,   # Vert: Chemin
        #        (0, 0, 0): 3,     # Noir: Robot
        #        (255, 255, 255): 2  # Blanc: Objectif
        #    }
        #encoded_image = np.zeros((visuelopti.shape[0], visuelopti.shape[1]), dtype=np.int32)
        #for color, value in colors.items():
        #    mask = np.all(visuelopti == np.array(color), axis=-1)
        #    encoded_image[mask] = value
        #self.visuelopti = encoded_image
        #self.base = np.copy(self.visuelopti)
        #self.objectif_position = objectif

    """
    def draw_basic_map(self) :
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
        self.base = image
        self.objectif = cercle_centre
        self.visuel = image
    """

    def __str__(self):
        return "L'environnement comporte un agent : "+str(self.agent)
        
     # Normalement jamais appelé   
    def redraw(self):
        return None
    
        if self.agent != None and self.visuel!=None :
            draw = ImageDraw.Draw(self.visuel)
            triangle_points = [self.agent.top_point, self.agent.bottom_right, self.agent.bottom_left]
            draw.polygon(triangle_points, fill=(0, 0, 0))
    """      
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


            if (0, 0, 0) in colors:
                robot_color_index = list(colors.keys()).index((0, 0, 0))
                robot_mask = masks[:, :, robot_color_index]
                if np.any(robot_mask):
                    robot_speed = self.agent.vitesse
                    robot_orientation = self.agent.orientation
                    processed_image[:, :, 4][robot_mask] += robot_speed
                    processed_image[:, :, 5][robot_mask] += robot_orientation

            self.state = processed_image
    """
    

    def reset_env(self):
        #self.visuel = draw_basic_map()
        self.agent.replace_agent(np.array([390, 150], dtype=np.int16))

    def compute_reward(self):
        #start_time = time.time()

        robot_point = self.agent.position
        agent_info = self.map[int(robot_point[0]), int(robot_point[1])]

        #indices_obj = np.argwhere(self.state[:, :, 2] == 4)
        #indices_obj = indices_obj[:, ::-1]
        #print(indices_obj)
        #distances = np.sqrt(np.abs(robot_point[0] - indices_obj[0, :])**2 + np.abs(robot_point[1]-indices_obj[1, :])**2)
        #distance = np.linalg.norm(self.objectif - robot_point, axis=1)
        #print("objectif : ",self.objectif)
        #print("robot : ", self.agent.position)
        distance = 2*np.sum(((self.objectif - self.agent.position)/400)**2,  axis=0)
        #print("distance : ", distance)
        #distance = np.mean(distances)
        """
        print(robot_point, self.objectif)
        dist1 = robot_point[0]-self.objectif[0]
        dist2 = robot_point[1]-self.objectif[1]
        dist1_carre = np.square(dist1)
        dist2_carre = np.square(dist2)
        somme = dist1_carre+dist2_carre
        distance = np.sqrt(somme)
        print(dist1, dist2)
        print(dist1_carre, dist2_carre)
        
        """
        #difference = self.previous_distance - distance
        #reward = -distance
        
        reward = 1-distance
        # Si première composante de 3ème dimension = 1 : rouge alors on pénalise
        if agent_info == 0:
            reward -= 10 
        
        
        self.previous_distance = distance
        return reward
       
    
    def step(self, action):
        #start_time = time.time()
        # Action est un tuple : action 0 = rotation  action 1 = vitesse   (premier exemple avec 4 modalités chacune)
        previous_state = self.map
        previous_agent = [self.agent.position[0], self.agent.position[1], self.agent.vitesse, self.agent.orientation]


        self.agent.rotate_robot(action[0])
        #self.agent.rotate_robot(action)
        self.agent.changer_vitesse(action[1])
        self.agent.avancer()
        self.agent.position = np.clip(self.agent.position, 0, 399)
        reward = self.compute_reward()

        next_agent = np.array([self.agent.position[0], self.agent.position[1], self.agent.vitesse, self.agent.orientation])
        #print(next_agent, "next agent")

        new_state = self.map
        #print("distance restante : ", self.previous_distance)
        if self.map[self.agent.position[0], self.agent.position[1]]!=2 :
            done=0
        else :
            done=1
            reward +=100
        #print("en résultat à cela, done = ", done)

        episode = [previous_state, action, reward, new_state, done, previous_agent, next_agent]
        self.agent.remember(episode)
        # Retour pas nécéssaire 
        #end_time = time.time()
        #execution_time = end_time - start_time
        #print("Temps d'exécution : {:.2f} secondes".format(execution_time))
        return reward, done