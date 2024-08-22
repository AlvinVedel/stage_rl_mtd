import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
import matplotlib.colors as mcolors
import math 
LARGEUR_MAX = 10
LARGEUR_MIN = 5
gamma = 0.95

def fill_circle(map, x, r, v=1):
    X, Y = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))
    distance = (X - x[0])**2 + (Y - x[1])**2
    map[distance <= r**2] = v

def bezier_curve(p0, p2, n_points=200):
    if(np.random.rand()<0.5):
      p1 = np.random.uniform(0.4, 0.6, size=(2,))
    else:
      p1 = np.array([0,0])
    t = np.linspace(0, 1, n_points)
    x = (1-t)**1 * p0[0] + 2*(1-t)*t*p1[0] + t**1 * p2[0]
    y = (1-t)**1 * p0[1] + 2*(1-t)*t*p1[1] + t**1 * p2[1]
    return x, y

def genMap(N, npoints):
  #background = np.ones((N*2+10, N*2+10), dtype=np.int8)*5
  map = np.zeros((N, N))

  for _ in range(npoints):
    pA = np.random.randint(0, N-1,  size=2)
    pB = np.random.randint(0, N-1,  size=2)
    pC = np.random.randint(LARGEUR_MIN, LARGEUR_MAX,  size=1)[0]

    fill_circle(map, pA, pC)
    x, y = bezier_curve(pA, pB, n_points=200)
    for i in range(x.shape[0]):
      xx, yy = x[i], y[i]
      pC = min(max(LARGEUR_MIN, np.random.randint(pC-2, pC+2,  size=1)[0]),LARGEUR_MAX )
      fill_circle(map, np.array([xx, yy]), pC)



  return map


actionSet = {0 :[-1, -1],
             1 : [-1, 0],
             2: [-1, 1],
             3 : [0, -1],
             4 :[0, 0],
             5 : [0, 1],
             6: [1, -1],
             7 :[1, 0],
             8 : [1, 1]}



class GridEnv(gym.Env):
    def __init__(self, grid_size, N,  NWallMax):
        super(GridEnv, self).__init__()
        self.toShow=False
        self.grid_size = grid_size
        self.N=N
        self.iter = 0
        self.succeed = 0
        self.success=False

    def tirage_objectif(self, distance_max):
        while True:
            angle = np.random.uniform(0, 2*np.pi)
            distance = np.random.uniform(3, distance_max)
            x = self.init_pos[0] +distance*np.cos(angle)
            y = self.init_pos[1] + distance*np.sin(angle)
            if 1<= x < self.grid_size-1 and 1<=y<self.grid_size-1 :
                return np.array([x, y]).astype(int)

    def reset(self,inference=False, seed=None, ):
        if(seed!=None):
            np.random.seed(10+seed)
            random.seed(10+seed)
        
        
        if self.success or self.iter==0:
            self.map = genMap(self.grid_size, 5)   
            self.init_pos = np.random.randint(2, self.grid_size-4, size=2)*1.0
            if self.iter < 800 :
                self.objectif = self.tirage_objectif(3+(self.iter+1)/40)
            else :
                self.objectif = np.random.randint(2, self.grid_size-4, 2)
            self.map[self.objectif[0], self.objectif[1]] = 2
        elif inference :
            self.map = genMap(self.grid_size, 5)   
            self.init_pos = np.random.randint(2, self.grid_size-4, size=2)*1.0
            self.objectif = np.random.randint(2, self.grid_size-4, 2)
            self.map[self.objectif[0], self.objectif[1]] = 2

        self.position = self.init_pos
        self.d_init = np.sqrt(np.sum(np.power(self.objectif-self.position, 2), axis=0))
        self.passe = np.zeros((self.grid_size, self.grid_size))
        self.iter+=1

        self.vitesse  = 1
        self.orientation  = 0.0
        self.success = False

        return self._get_observation()

    def rotate_robot(self, direction):
            self.orientation += direction*(np.pi/10)  
            self.orientation %= 2*np.pi

    def changer_vitesse(self, vitesse):
        self.vitesse = np.clip( (vitesse*0.05)+self.vitesse , -1, 5)                                                             
                
    def avancer(self, agent_info):
        denominateur = np.max([4*(1-np.min([1, agent_info])), 1]) # SI IL EST SUR DU ROUGE   -> si rouge : 0 alors  max entre 4 * (1 - 0) et 1 => 4 ai dénominateur
        delta_x = (self.vitesse/denominateur) * np.cos(self.orientation)  
        delta_y = (self.vitesse/denominateur) * np.sin(self.orientation) 
            
        self.position += np.array([delta_x, delta_y], dtype=np.float32) 
        #self.position = np.nan_to_num(self.position, nan=0)
        self.position=np.clip(self.position, 0, self.grid_size-1)

    def step(self, action):#0...8
        action=action[0]
        action_rotate, action_vitesse = actionSet[action]
        self.rotate_robot(action_rotate)
        self.changer_vitesse(action_vitesse)

        type_case = self.map[np.clip(int(self.position[0]), 0, self.grid_size-1), np.clip(int(self.position[1]), 0, self.grid_size-1)]
        self.avancer(type_case)
        #print(self.position, self.objectif, self.vitesse, self.orientation,action)
        meurt=False
        estpasse = self.passe[int(self.position[0]), int(self.position[1])]
        self.passe[int(self.position[0]), int(self.position[1])]=1
        reward  = (-0.1) * (1 +estpasse)
        if(np.sum((self.position-self.objectif)**2)<3):
            reward=10
            meurt=True
            self.success=True
            self.succeed+=1

        return self._get_observation(), reward, meurt, ""




    def _get_observation(self):

        liste_case = []
        liste_case_p = []
        angles = np.arange(self.orientation-(np.pi/6), self.orientation+(np.pi/6)+(np.pi/30), np.pi/30)[:11]
        distances = np.arange(1, 20, 1)
        liste_coord = []
        for distance in distances :
            for angle in angles :
                x = int(self.position[0]+distance*np.cos(angle))
                y = int(self.position[1]+distance*np.sin(angle))
                if x > 0 and x < self.grid_size-1 and y > 0 and y < self.grid_size-1 :
                    new_case = self.map[x, y]
                    new_case_p = self.passe[x, y]
                    
                    if (x, y) not in liste_coord :
                        liste_coord.append((x, y))
                else :
                    new_case = 3
                    new_case_p = 0
                liste_case.append(new_case)
                liste_case_p.append(new_case_p)
        liste_case = np.array(liste_case, dtype=np.int8)
        liste_case_p = np.array(liste_case_p, dtype=np.int8)

        toto = np.zeros((3))
        toto[0]= np.arctan2(self.position[1]-self.objectif[1], self.position[0] - self.objectif[0])
        toto[1]= self.vitesse
        toto[2]=self.orientation


        liste_obs=[liste_case,toto, liste_case_p, liste_coord]        

        return [liste_obs]

