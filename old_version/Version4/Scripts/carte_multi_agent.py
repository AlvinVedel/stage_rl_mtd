import numpy as np
from PIL import Image
from alvin_env import *


seed = 664746509
np.random.seed(seed)
random.seed(seed)

def fill_circle(map, x, r, v=1):
    X, Y = np.meshgrid(np.arange(map.shape[0]), np.arange(map.shape[1]))
    distance = (X - x[0])**2 + (Y - x[1])**2
    map[distance <= r**2] = v

def obstacle(map, x, r, v=3):
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

def genMap(N, npoints, nobstacles):
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
  for _ in range(nobstacles):
    pO = np.random.randint(7, N-8, size=2)
    obstacle(map, pO, 5)
  return map



env = GridEnv(grid_size=100, N=1, NWallMax=40)

env.map = genMap(env.grid_size, 5, 0)
env.position = np.array([22, 33])
env.objectif = np.array([76, 71])
env.map[env.objectif[0], env.objectif[1]] = 2


base2 = np.zeros((100, 100, 3), dtype=np.uint8)
base2[np.where(env.map == 0)] = [255, 0, 0]
base2[np.where(env.map == 1)] = [0, 255, 0]
base2[np.where(env.map==3)] = [255, 165, 0]        
base2[int(env.position[0]), int(env.position[1])] = [0, 0, 0]
print(env.position)


"""
if len(states[0][3])>0 :
    x_coords, y_coords = zip(*states[0][3])
    base2[np.array(x_coords), np.array(y_coords)] = [255, 255, 0]
"""
base2[np.where(env.map == 2)] = [255, 255, 255]
visuel2 = base2.copy()

visuel2 = Image.fromarray(visuel2)
visuel2.save("test_carte.png")