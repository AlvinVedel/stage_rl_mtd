#!/usr/bin/env python
# coding: utf-8

import yaml
import numpy as np
import math as math
import random as random
import pygame
import time
from PIL import Image

# Conversion des unités
km_per_hour_to_m_per_sec = 1000 / 3600
km_to_pixels = 140.5760 / 0.1384  # pixels par kilomètre

# Définir la vitesse maximale en km/h
max_speed_km_h = 30  # 30 km/h

# Convertir la vitesse maximale en pixels par seconde
max_speed_m_per_sec = max_speed_km_h * km_per_hour_to_m_per_sec
max_speed_pixels_per_sec = max_speed_m_per_sec * km_to_pixels

# Définir une plage de vitesse réaliste (par exemple, entre 10% et 100% de la vitesse maximale)
min_speed_pixels_per_sec = max_speed_pixels_per_sec * 0.1

'''

  In this file, you must write all the functions specific to your environment.

'''
# Définition des couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
GRAY = (200, 200, 200)
ORANGE = (255, 127, 0)
PURPLE = (128, 0, 128)
PINK = (218, 112, 214)
BGRD = (51, 128, 255)
CIRCUIT = (51, 178, 255)

class PygameSingleton:
    _instance = None
    _initialized = False
    _screen = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PygameSingleton, cls).__new__(cls)
        return cls._instance

    def __init__(self, screen_width, screen_height):
        if not self._initialized:
            pygame.init()
            self._screen = pygame.display.set_mode((screen_width, screen_height))
            self._initialized = True

    @property
    def screen(self):
        return self._screen
 

def check_observation_limits(state):
    state_np = np.array(state)
    print(f"Min value in state: {np.min(state_np)}")
    print(f"Max value in state: {np.max(state_np)}")

def normalize_observation(state, min_val=-318806.34, max_val=318806.34):
    state_np = np.array(state, dtype=np.float32)
    # Normalisation min-max
    state_normalized = (state_np - np.min(state_np)) / (np.max(state_np) - np.min(state_np)) * (max_val - min_val) + min_val
    return state_normalized

'''
def get_observations(data, ground_window, size=42):
    
    state = [[255 for _ in range(size)] for _ in range(size)]
     
    for j in range(size):
        state[0][j] = data[j]
    
    
    for row in range(len(ground_window)):
        for col in range(len(ground_window[0])):
            state[row + 1][col + 1] = ground_window[row][col]
   
    state_np = np.array(state)
    print(f"Min value in state: {np.min(state_np)}")
    print(f"Max value in state: {np.max(state_np)}")
    
    
    return state    
'''

def get_red_component_window(image, boat_x, boat_y, pixels_per_meter, window_size=42, center_square_size=4):
    # Conversion des coordonnées du bateau
    pos_x, pos_y = ned_to_image(boat_x, boat_y, pixels_per_meter)

    # Charger l'image
    img = Image.open(image)
    img = img.convert('RGB')  # Assurer que l'image est en mode RGB
    img_array = np.array(img)

    # Dimensions de l'image
    img_height, img_width, _ = img_array.shape

    # Initialiser la fenêtre de 42x42 avec la valeur 255 (couleur rouge (255, 0, 0))
    red_window = np.full((window_size, window_size), 255, dtype=np.uint8)

    # Vérifier que pos_x et pos_y sont à l'intérieur de l'image
    if pos_x < 0 or pos_x >= img_width or pos_y < 0 or pos_y >= img_height:
        return red_window

    # Taille de la demi-fenêtre
    half_window = window_size // 2

    # Calculer les limites de la fenêtre dans l'image
    start_x = max(int(pos_x) - half_window, 0)
    end_x = min(int(pos_x) + half_window + 1, img_width)
    start_y = max(int(pos_y) - half_window, 0)
    end_y = min(int(pos_y) + half_window + 1, img_height)

    # Vérifier que les dimensions de la fenêtre dans l'image sont valides
    if start_x >= end_x or start_y >= end_y:
        return red_window

    # Calculer les indices de la fenêtre locale où les valeurs doivent être copiées
    local_start_x = max(half_window - (int(pos_x) - start_x), 0)
    local_start_y = max(half_window - (int(pos_y) - start_y), 0)
    local_end_x = local_start_x + (end_x - start_x)
    local_end_y = local_start_y + (end_y - start_y)

    # Copier les valeurs de la composante rouge de l'image dans la fenêtre
    red_window[local_start_y:local_end_y, local_start_x:local_end_x] = img_array[start_y:end_y, start_x:end_x, 0]

    # Positionner le bateau dans la fenêtre
    # Déterminer les indices de début et de fin pour le carré central
    center_start = window_size // 2 - center_square_size // 2
    center_end = center_start + center_square_size

    # Placer le carré central contenant la valeur 100
    red_window[center_start:center_end, center_start:center_end] = 100
    
    
    print("len(red_window)=",len(red_window))

    return red_window
    
    
def get_red_component_window2(image, boat_x, boat_y, pixels_per_meter, window_size=42):


    pos_x, pos_y = ned_to_image(boat_x, boat_y, pixels_per_meter)

    # Charger l'image
    img = Image.open(image)
    img = img.convert('RGB')  # Assurer que l'image est en mode RGB
    img_array = np.array(img)

    # Dimensions de l'image
    img_height, img_width, _ = img_array.shape
    
     # Initialiser la fenêtre de 41x41 avec la valeur 255 (couleur rouge (255, 0, 0))
    red_window = np.full((window_size, window_size), 255, dtype=np.uint8)

    # Vérifier que pos_x et pos_y sont à l'intérieur de l'image
    if pos_x < 0 or pos_x >= img_width or pos_y < 0 or pos_y >= img_height:
        return red_window

    # Taille de la demi-fenêtre
    half_window = window_size // 2

    # Calculer les limites de la fenêtre dans l'image
    start_x = max(int(pos_x) - half_window, 0)
    end_x = min(int(pos_x) + half_window + 1, img_width)
    start_y = max(int(pos_y) - half_window, 0)
    end_y = min(int(pos_y) + half_window + 1, img_height)
    

    # Vérifier que les dimensions de la fenêtre dans l'image sont valides
    if start_x >= end_x or start_y >= end_y:
        return red_window

    
    # Calculer les indices de la fenêtre locale où les valeurs doivent être copiées
    local_start_x = max(half_window - (int(pos_x) - start_x), 0)
    local_start_y = max(half_window - (int(pos_y) - start_y), 0)
    local_end_x = local_start_x + (end_x - start_x)
    local_end_y = local_start_y + (end_y - start_y)

    # Copier les valeurs de la composante rouge de l'image dans la fenêtre
    red_window[local_start_y:local_end_y, local_start_x:local_end_x] = img_array[start_y:end_y, start_x:end_x, 0]
    
    # Positionner le bateau dans la fenêtre
    # Déterminer les indices de début et de fin pour le carré 3x3
    center_start = window_size // 2 - 3 // 2
    center_end = center_start + 3

    # Placer le carré 5x5 contenant la valeur 100 au centre
    red_window[center_start:center_end, center_start:center_end] = 100
    
    

    return red_window



def get_concurrent_image_position(boat_position, boat_target_angle, distance, detection_angle, pixels_per_meter):
    # Convertir les angles de degrés en radians
    target_angle_radians = math.radians(boat_target_angle + detection_angle)
       
    # Calculer les coordonnées cartésiennes
    x_ned = boat_position[0] + distance * math.cos(target_angle_radians)
    y_ned = boat_position[1] + distance * math.sin(target_angle_radians)
    
    pos_x, pos_y = ned_to_image(x_ned, y_ned, pixels_per_meter)
    
    concurrent_image_position = [pos_x, pos_y]
    
    return concurrent_image_position
    

def get_sitac_window(image, boat_x, boat_y, boat_target_angle, pixels_per_meter, detection_ranges, directions, detected_sectors_per_range, window_size=42, center_square_size=4):

     # Conversion des coordonnées du bateau
    pos_x, pos_y = ned_to_image(boat_x, boat_y, pixels_per_meter)

    # Charger l'image
    img = Image.open(image)
    img = img.convert('RGB')  # Assurer que l'image est en mode RGB
    img_array = np.array(img)

    # Dimensions de l'image
    img_height, img_width, _ = img_array.shape

    # Initialiser la fenêtre de 42x42 avec la valeur 255 (couleur rouge (255, 0, 0))
    red_window = np.full((window_size, window_size), 255, dtype=np.uint8)

    # Vérifier que pos_x et pos_y sont à l'intérieur de l'image
    if pos_x < 0 or pos_x >= img_width or pos_y < 0 or pos_y >= img_height:
        return red_window

    # Taille de la demi-fenêtre
    half_window = window_size // 2

    # Calculer les limites de la fenêtre dans l'image
    start_x = max(int(pos_x) - half_window, 0)
    end_x = min(int(pos_x) + half_window, img_width - 1)
    start_y = max(int(pos_y) - half_window, 0)
    end_y = min(int(pos_y) + half_window, img_height - 1)

    # Vérifier que les dimensions de la fenêtre dans l'image sont valides
    if start_x >= end_x or start_y >= end_y:
        return red_window

    # Calculer les indices de la fenêtre locale où les valeurs doivent être copiées
    local_start_x = half_window - (int(pos_x) - start_x)
    local_start_y = half_window - (int(pos_y) - start_y)
    local_end_x = local_start_x + (end_x - start_x)
    local_end_y = local_start_y + (end_y - start_y)

    # Copier les valeurs de la composante rouge de l'image dans la fenêtre
    red_window[local_start_y:local_end_y, local_start_x:local_end_x] = img_array[start_y:end_y, start_x:end_x, 0]
    
    # Positionner le bateau dans la fenêtre
    # Déterminer les indices de début et de fin pour le carré central
    center_start = window_size // 2 - center_square_size // 2
    center_end = center_start + center_square_size

    # Placer le carré central contenant la valeur 100
    red_window[center_start:center_end, center_start:center_end] = 100
    
    
    
    boat_position = [boat_x, boat_y]
    
    # Taille du carré à modifier
    square_size = 3
    half_square = square_size // 2
    
    for range_num in range(len(detection_ranges)):
        
        for sector_num in range(len(directions)):
            
            if detected_sectors_per_range[range_num][sector_num] == 1:
            
                distance = detection_ranges[range_num]
    
                direction = directions[sector_num]
                
                concurrent_image_position = get_concurrent_image_position(boat_position, 
                                                                    boat_target_angle, 
                                                                    distance, 
                                                                    direction, 
                                                                    pixels_per_meter)
            
                # Parcourir les coordonnées du carré autour de (global_x, global_y)
                for dx in range(-half_square, half_square + 1):
                    for dy in range(-half_square, half_square + 1):
                        # Convertir les coordonnées globales en coordonnées locales relatives à red_window
                        local_x = concurrent_image_position[0] + dx - start_x + local_start_x
                        local_y = concurrent_image_position[1] + dy - start_y + local_start_y

                        # Vérifier que les coordonnées locales sont dans les limites de red_window
                        if 0 <= local_x < window_size and 0 <= local_y < window_size:
                            # Modifier la valeur dans red_window
                            red_window[local_y, local_x] = 255
                                
                # Convertir les coordonnées globales en coordonnées locales relatives à red_window
                # local_x = concurrent_image_position[0] - start_x + local_start_x
                # local_y = concurrent_image_position[1] - start_y + local_start_y

                # Vérifier que les coordonnées locales sont dans les limites de red_window
                # if 0 <= local_x < window_size and 0 <= local_y < window_size:
                    # Modifier la valeur dans red_window
                    # red_window[local_y, local_x] = 255
                
                
    return red_window

def check_if_outside_track_limit_or_concurrents_collision(ground_window, window_size=35):  

    choice_goes_to_collision = False
    
    # Calculer les coordonnées centrales de ground_window
    center_x = window_size // 2
    center_y = window_size // 2

    # Taille du carré à accéder
    square_size = 3
    half_square = square_size // 2

    # Parcourir les coordonnées du carré autour du centre de ground_window
    for dx in range(-half_square, half_square + 1):
        for dy in range(-half_square, half_square + 1):
            # Calculer les coordonnées locales relatives à ground_window
            local_x = center_x + dx
            local_y = center_y + dy

            # Vérifier que les coordonnées locales sont dans les limites de ground_window
            if 0 <= local_x < window_size and 0 <= local_y < window_size:
                # Accéder à la valeur dans ground_window
                value = ground_window[local_y, local_x]
                
                if value == 255:
                    choice_goes_to_collision = True
                    break
  
    return choice_goes_to_collision 
    
    
def get_red_component_at_position(image, pos_x, pos_y):
    # Charger l'image
        
    img = Image.open(image)
    img = img.convert('RGB')  # Assurer que l'image est en mode RGB
    img_array = np.array(img)

    # Dimensions de l'image
    img_height, img_width, _ = img_array.shape

    # Vérifier que les coordonnées sont dans les limites de l'image
    if pos_x < 0 or pos_x >= img_width or pos_y < 0 or pos_y >= img_height:
        red_component = 255

    # Récupérer la composante rouge à la position spécifiée
    red_component = img_array[pos_y, pos_x, 0]

    return red_component


def latlon_to_meters(lat1, lon1, lat2, lon2, earth_radius):
    """Convertit les différences de latitude et longitude en mètres."""
    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)
    lat1_rad = np.radians(lat1)

    delta_x = delta_lon * earth_radius * np.cos(lat1_rad)
    delta_y = delta_lat * earth_radius
    return delta_x, delta_y

def calculate_transform_params(x1, y1, lat1, lon1, x2, y2, lat2, lon2, earth_radius):
    """Calcule les paramètres de transformation pour conversion entre l'image et les coordonnées géographiques."""
    delta_x_meters, delta_y_meters = latlon_to_meters(lat1, lon1, lat2, lon2, earth_radius)
    
    delta_x_pixels = x2 - x1
    delta_y_pixels = y2 - y1
    
    scale_x = delta_x_meters / delta_x_pixels
    scale_y = delta_y_meters / delta_y_pixels
    
    return scale_x, scale_y, x1, y1, lat1, lon1

def image_to_latlon(x, y, scale_x, scale_y, x_origin, y_origin, lat_origin, lon_origin, earth_radius):
    """Convertit les coordonnées de l'image en coordonnées géographiques."""
    delta_x = (x - x_origin) * scale_x
    delta_y = (y - y_origin) * scale_y
    
    lat = lat_origin + (delta_y / earth_radius) * (180 / np.pi)
    lon = lon_origin + (delta_x / (earth_radius * np.cos(np.radians(lat_origin)))) * (180 / np.pi)
    
    return lat, lon

def latlon_to_image(lat, lon, scale_x, scale_y, x_origin, y_origin, lat_origin, lon_origin, earth_radius):
    """Convertit les coordonnées géographiques en coordonnées de l'image."""
    delta_x, delta_y = latlon_to_meters(lat_origin, lon_origin, lat, lon, earth_radius)
    
    x = x_origin + delta_x / scale_x
    y = y_origin + delta_y / scale_y
    
    return int(x), int(y)
    
# Conversion des sommets en coordonnées NED
def image_to_ned(x, y, pixels_per_meter, x_origin=0, y_origin=0):
    delta_x = x - x_origin
    delta_y = y - y_origin
    north_y = -delta_y / pixels_per_meter
    east_x = delta_x / pixels_per_meter
    return east_x, north_y 

def ned_to_image(east_x, north_y, pixels_per_meter, image_width=1280, image_height=720):
    x_origin = 0
    y_origin = 0
    x = east_x * pixels_per_meter + x_origin
    y = -north_y * pixels_per_meter + y_origin
    return int(x), int(y)
    
def degrees_to_radians(degrees):
    return degrees * (math.pi / 180.0)

def polar_to_ned(distance, direction_degrees):
    direction_radians = degrees_to_radians(direction_degrees)
    north_y = distance * math.cos(direction_radians)
    east_x = distance * math.sin(direction_radians)
    return east_x, north_y 
    
# Fonction pour obtenir la position NED d'une case donnée
def get_ned_position(boat_target_angle, range_index, direction_index, detection_ranges, directions):
    
    distance = detection_ranges[range_index]
    
    direction = directions[direction_index]
    
    east_x, north_y = polar_to_ned(distance, boat_target_angle + direction)
    
    return (east_x, north_y) 


def ned_to_polar(east, north):
    distance = math.sqrt(north**2 + east**2)
    direction_radians = math.atan2(east, north)
    direction_degrees = math.degrees(direction_radians)
    return distance, direction_degrees      

def load_values_from_yaml(filename, main_key, object_key):
    # Lire le fichier YAML
    with open(filename, 'r') as file:
        data = yaml.safe_load(file)

    # Accéder au tableau de valeurs
    val = data[main_key][object_key]

    return val

def load_quadrilaterals_from_yaml(filename, main_key, object_key):
    with open(filename, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    # Extraire les quadrilatères
    quadrilaterals = data[main_key][object_key]

    # Convertir en une liste de listes de tuples (x, y)
    track_quadrilaterals = []
    for quad in quadrilaterals:
        vertices = [(vertex['vertex']['x'], vertex['vertex']['y']) for vertex in quad['quadrilateral']]
        track_quadrilaterals.append(vertices)

    return track_quadrilaterals


def load_bouees_from_yaml(filename, main_key, object_key):
    with open(filename, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    points = []
    
    centers = data.get(main_key, {}).get(object_key, [])
    for center in centers:
        point = (center['center']['x'], center['center']['y'])
        points.append(point)

    return points

def load_polygon_from_yaml(filename, main_key, object_key):
    with open(filename, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
    
    points = []
    
    vertices = data.get(main_key, {}).get(object_key, [])
    for vertex in vertices:
        point = (vertex['vertex']['x'], vertex['vertex']['y'])
        points.append(point)

    return points


def calculate_midpoint(x1, y1, x2, y2):
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    return x_mid, y_mid

def calculate_distance(x1, y1, x2, y2):
    """
    Calcule la distance entre deux points dans un espace bidimensionnel.
    
    Args:
    x1 (float): Coordonnée x du premier point.
    y1 (float): Coordonnée y du premier point.
    x2 (float): Coordonnée x du deuxième point.
    y2 (float): Coordonnée y du deuxième point.
    
    Returns:
    float: La distance entre les deux points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    """
    Calcule l'angle entre la droite formée par deux points et l'axe x.
    
    Args:
    x1 (float): Coordonnée x du premier point.
    y1 (float): Coordonnée y du premier point.
    x2 (float): Coordonnée x du deuxième point.
    y2 (float): Coordonnée y du deuxième point.
    
    Returns:
    float: L'angle en radians entre la droite formée par les deux points et l'axe x.
    """
    dx = x2 - x1
    dy = y2 - y1
    return math.atan2(dy, dx)    
    
def max_direction_change(speed, min_direction_change, max_direction_change_base):
    # Moduler le changement de direction maximal en fonction de la vitesse
    # Plus la vitesse est grande plus le changement de direction est minime
    return max(min_direction_change, max_direction_change_base - (speed / 30.0) * (max_direction_change_base - min_direction_change))

    
def limite_de_temps_atteinte(current_time, start_time, max_race_time, time_scaling_factor):
    elapsed_time = (current_time - start_time) * time_scaling_factor
    return elapsed_time >= max_race_time
    
    
def calculate_battery_consumption(initial_battery_level, speed, base_consumption_rate, speed_consumption_factor):
    """
    Calculate the battery consumption based on speed and base consumption rate.
    """
    
    battery_consumption = base_consumption_rate + (speed_consumption_factor * speed)
    
    battery_level = initial_battery_level - battery_consumption
    if battery_level < 0:
        battery_level = 0
    
    return battery_level
    
def better_lap_time(current_lap_time, best_lap_time):
    return current_lap_time < best_lap_time
    
def low_battery(battery_level, critical_level):
    return battery_level <= critical_level

"""
def get_gateways(big_hexagon_points, small_hexagon_points):
    # Liste pour stocker les coordonnées des couples de points reliant les sommets
    gateways = []

    for i in range(6):
        # Stocker les coordonnées des couples de points reliant les sommets
        gateways.append((big_hexagon_points[i], small_hexagon_points[i]))
    
    return gateways
"""

def get_gateways(big_hexagon_points, small_hexagon_points):
    
    gateways = []
    radiuses = np.zeros(6)

    for i in range(6):
        exter = np.array([big_hexagon_points[i][0], big_hexagon_points[i][1]])
        inter = np.array([small_hexagon_points[i][0], small_hexagon_points[i][1]])
        middle = (inter+exter)/2
        radius = np.sqrt(np.sum((exter - inter)**2)) / 2
        gateways.append(middle.astype(np.int32))
        radiuses[i] =radius
    
    return gateways, radiuses

def is_same_gateways(gateway_one, gateway_two):
    val = False
    
    if gateway_one[0][0] == gateway_two[0][0] and gateway_one[0][1] == gateway_two[0][1] and gateway_one[1][0] == gateway_two[1][0] and gateway_one[1][1] == gateway_two[1][1]:
        val = True
    return val

def distance_to_gateway(x, y, xa, ya, xb, yb):
    # Calculate the distance to the infinite line as before
    distance = np.abs((yb - ya) * x - (xb - xa) * y + xb * ya - yb * xa) / np.sqrt((yb - ya) ** 2 + (xb - xa) ** 2)
    
    # Vector from point A to point B
    AB = np.array([xb - xa, yb - ya])
    # Vector from point A to the point (x, y)
    AP = np.array([x - xa, y - ya])
    # Vector from point B to the point (x, y)
    BP = np.array([x - xb, y - yb])

    # Project vector AP onto AB to find the closest point on the line segment
    proj = np.dot(AP, AB) / np.dot(AB, AB)

    if proj < 0:
        # Closest point is A
        closest_point = (xa, ya)
    elif proj > 0 and proj < 1:
        # Closest point is within the segment
        closest_point = (xa + proj * AB[0], ya + proj * AB[1])
    else:
        # Closest point is B
        closest_point = (xb, yb)

    # If the closest point on the infinite line is outside the segment,
    # the distance to the segment is the distance to the nearest endpoint
    if proj < 0 or proj > 1:
        distance = min(np.sqrt(np.sum(AP**2)), np.sqrt(np.sum(BP**2)))

    return distance
    
def normalized_distance_to_gateway(_x, _y, _xa, _ya, _xb, _yb, image_width, image_height, pixels_per_meter):

    x, y = ned_to_image(_x, _y, pixels_per_meter, image_width, image_height)
    xa, ya = ned_to_image(_xa, _ya, pixels_per_meter, image_width, image_height)
    xb, yb = ned_to_image(_xb, _yb, pixels_per_meter, image_width, image_height)

    # Normaliser les coordonnées
    x_norm = x / image_width
    y_norm = y / image_height
    xa_norm = xa / image_width
    ya_norm = ya / image_height
    xb_norm = xb / image_width
    yb_norm = yb / image_height

    # Calculer la distance normalisée à la ligne infinie
    distance = np.abs((yb_norm - ya_norm) * x_norm - (xb_norm - xa_norm) * y_norm + xb_norm * ya_norm - yb_norm * xa_norm) / np.sqrt((yb_norm - ya_norm) ** 2 + (xb_norm - xa_norm) ** 2)

    # Vecteurs pour la projection
    AB = np.array([xb_norm - xa_norm, yb_norm - ya_norm])
    AP = np.array([x_norm - xa_norm, y_norm - ya_norm])
    BP = np.array([x_norm - xb_norm, y_norm - yb_norm])

    # Projection de AP sur AB
    proj = np.dot(AP, AB) / np.dot(AB, AB)

    # Déterminer le point le plus proche sur le segment
    if proj < 0:
        closest_point = (xa_norm, ya_norm)
    elif proj > 1:
        closest_point = (xb_norm, yb_norm)
    else:
        closest_point = (xa_norm + proj * AB[0], ya_norm + proj * AB[1])

    # Calculer la distance au segment
    if proj < 0 or proj > 1:
        distance = min(np.sqrt(np.sum(AP**2)), np.sqrt(np.sum(BP**2)))

    return distance
    
def position_relative_to_gateway(x, y, xa, ya, xb, yb):
    return np.sign((xb - xa) * (y - ya) - (yb - ya) * (x - xa))


def max_direction_change(speed, min_direction_change, max_direction_change_base):
    # Moduler le changement de direction maximal en fonction de la vitesse
    return max(min_direction_change, max_direction_change_base - (speed / 30.0) * (max_direction_change_base - min_direction_change))
   
def nearest_value_index(values, target):
    # Trouve la valeur et l'index les plus proches dans une liste
    index = min(range(len(values)), key=lambda i: abs(values[i] - target))
    return index
        

def action_to_range_and_direction_indexes(action, 
                        current_speed, 
                        direction_intentions, 
                        speed_intentions,                        
                        directions, 
                        speeds,
                        detection_ranges,
                        max_speed_change,
                        min_direction_change, 
                        max_direction_change_base):
                        
    # Décoder l'action en intentions de direction et de vitesse
    direction_action = action // len(direction_intentions)
    speed_action = action % len(speed_intentions)

    # Appliquer les intentions de direction
    direction = direction_intentions[direction_action] * max_direction_change(current_speed, min_direction_change, max_direction_change_base) / 2  # Modulation du changement de direction en fonction de la vitesse

    # Trouver l'index de la direction la plus proche
    direction_index = nearest_value_index(directions, direction) 
       
    # Déterminer la consigne de vitesse de l'agent
    min_speed = speeds[0]
    max_speed = speeds[len(speeds)-1]
    
    # Appliquer les intentions de vitesse 
    # -1: diminuer, 0: maintenir, 1: augmenter
    if speed_intentions[speed_action] == 1: # 1: augmenter
        speed = min(current_speed + max_speed_change, max_speed)
    elif speed_intentions[speed_action] == -1: # -1: diminuer
        speed = max(current_speed - max_speed_change, min_speed)
    elif speed_intentions[speed_action] == 0: # 0: maintenir
        speed = current_speed
        
    # Trouver l'index de la distance/vitesse la plus proche
    speed_index = nearest_value_index(speeds, speed)  
    range_index = speed_index
    
        
    return range_index, direction_index
    
    
def check_if_concurrents_collision(range_index, 
                                direction_index,
                                detected_sectors_per_range):  

    choice_goes_to_collision = False
    
    for range_num in range(range_index):                            
        if detected_sectors_per_range[range_num][direction_index] == 1:
            choice_goes_to_collision = True 
  
    return choice_goes_to_collision    
    

def get_new_current_boat_position(previous_boat_position, distance, target_angle_degrees):
    # Convertir les angles de degrés en radians
    target_angle_radians = math.radians(target_angle_degrees)
       
    # Calculer les coordonnées cartésiennes
    x = previous_boat_position[0] + distance * math.cos(target_angle_radians)
    y = previous_boat_position[1] + distance * math.sin(target_angle_radians)
    
    new_current_boat_position = [x,y]
    
    return new_current_boat_position



def check_is_inside_track(image, boat_current_pos, pixels_per_meter):

    is_inside_track = False
    
    x, y = ned_to_image(boat_current_pos[0], boat_current_pos[1], pixels_per_meter)
    
    red_component = get_red_component_at_position(image, int(x), int(y))
        
    if red_component == 0 :
        is_inside_track = True

    return is_inside_track 
    
    
    
def is_gateway_reached(boat_current_pos, gateway):
    is_reached = False

    # Calcul de la position du centre du segment
    center_x = (gateway[0][0] + gateway[1][0]) / 2
    center_y = (gateway[0][1] + gateway[1][1]) / 2
    center_gateway = (center_x, center_y)

    # Calcul de la demi-longueur du segment
    # Utilisation de np.linalg.norm pour la distance euclidienne
    half_length = np.linalg.norm(np.array(gateway[0]) - np.array(gateway[1])) / 2
        
    if np.linalg.norm(np.array(boat_current_pos) - np.array(center_gateway)) < half_length :
        is_reached = True

    return is_reached    
    


def is_gateway_passed(boat_x, boat_y, direction, speed, gateway, elapsed_time, num_subdivisions=1):
    is_passed = False

    # Calculer la distance parcourue pendant le temps écoulé
    scaled_speed = speed * elapsed_time

    # Déplacement vers le point de passage suivant
    newpos = (
        boat_x + scaled_speed * math.cos(direction),
        boat_y + scaled_speed * math.sin(direction)
    )
    
    
    # Position relative avant le déplacement
    gateway_relative_pos_before = np.sign(position_relative_to_gateway(
        boat_x, boat_y, 
        gateway[0][0], gateway[0][1], 
        gateway[1][0], gateway[1][1]
    ))

    # Position relative après le déplacement
    gateway_relative_pos_after = np.sign(position_relative_to_gateway(
        newpos[0], newpos[1], 
        gateway[0][0], gateway[0][1], 
        gateway[1][0], gateway[1][1]
    ))

    # Vérifier si la position relative a changé de signe
    if gateway_relative_pos_before != gateway_relative_pos_after:
        is_passed = True

    return is_passed


def check_track_limit_collision(boat_x_before, boat_y_before, boat_x_after, boat_y_after, agent_direction_choice, agent_speed_choice, quad, elapsed_time):
    collision = False
    
    # Calculer la distance parcourue pendant le temps écoulé
    # scaled_speed = agent_speed_choice * elapsed_time
    
    scaled_speed = agent_speed_choice

    # Calcul de la nouvelle position du bateau
    # newpos = (
        # boat_x + scaled_speed * math.cos(agent_direction_choice),
        # boat_y + scaled_speed * math.sin(agent_direction_choice)
    # )
    
    # Extraction des sommets du quadrilatère
    vertex_A1 = [quad[0][0], quad[0][1]]
    vertex_A2 = [quad[1][0], quad[1][1]]
    vertex_B1 = [quad[2][0], quad[2][1]]
    vertex_B2 = [quad[3][0], quad[3][1]]

    # Définition des segments du quadrilatère
    track_limit = [
        (vertex_A1, vertex_A2),
        (vertex_A2, vertex_B2),
        (vertex_B1, vertex_A1)
    ]

    # Segment représentant le mouvement du bateau
    boat_segment = ((boat_x_before, boat_y_before), (boat_x_after, boat_y_after))

    # Vérification des intersections
    for segment in track_limit:
        # if do_segments_intersect(boat_segment[0], boat_segment[1], segment[0], segment[1]):
            # collision = True
            # break
            
            
        # Position relative avant le déplacement
        gateway_relative_pos_before = np.sign(position_relative_to_gateway(
            boat_x_before, boat_y_before, 
            segment[0][0], segment[0][1], 
            segment[1][0], segment[1][1]
        ))

        # Position relative après le déplacement
        gateway_relative_pos_after = np.sign(position_relative_to_gateway(
            boat_x_after, boat_y_after, 
            segment[0][0], segment[0][1], 
            segment[1][0], segment[1][1]
        ))

        # Vérifier si la position relative a changé de signe
        if gateway_relative_pos_before != gateway_relative_pos_after:
            collision = True
            break       
    
    return collision

def do_segments_intersect(p1, p2, q1, q2):
    # Déterminer si les segments (p1, p2) et (q1, q2) se croisent
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0  # Collinear
        elif val > 0:
            return 1  # Clockwise
        else:
            return 2  # Counterclockwise

    def on_segment(p, q, r):
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # Cas général
    if o1 != o2 and o3 != o4:
        return True

    # Cas spéciaux
    if o1 == 0 and on_segment(p1, q1, p2):
        return True
    if o2 == 0 and on_segment(p1, q2, p2):
        return True
    if o3 == 0 and on_segment(q1, p1, q2):
        return True
    if o4 == 0 and on_segment(q1, p2, q2):
        return True

    return False




def required_laps_and_sufficient_battery_and_max_race_time(laps_completed, required_laps, battery_level, sufficient_battery_level, current_time, max_race_time, start_time, time_scaling_factor):
    
    elapsed_time = (current_time - start_time) * time_scaling_factor
    
    
    return (laps_completed >= required_laps and
            battery_level >= sufficient_battery_level and
            elapsed_time <= max_race_time)   
    
class QuadrilateralManager:
    def __init__(self, quadrilaterals):
        self.quadrilaterals = quadrilaterals
        self.current_quadrilateral_index = 0
        self.current_quadrilateral = self.quadrilaterals[self.current_quadrilateral_index]
        self.first_lap_completed = False

    def next_quadrilateral(self):
        if not self.first_lap_completed:
            if self.current_quadrilateral_index == 0:
                next_quadrilateral_index = 1
            else:
                next_quadrilateral_index = (self.current_quadrilateral_index + 1) % len(self.quadrilaterals)
                if next_quadrilateral_index == 1:
                    self.first_lap_completed = True
        else:
            next_quadrilateral_index = (self.current_quadrilateral_index + 1) % len(self.quadrilaterals)
            if next_quadrilateral_index == 0:
                next_quadrilateral_index = 1

        self.current_quadrilateral_index = next_quadrilateral_index
        self.current_quadrilateral = self.quadrilaterals[self.current_quadrilateral_index]
        
        
        
class PyBoat:
    def __init__(self, color, gateways, starting_zone_polygon_points, speeds, frames_per_second):
        self.color = color
        self.gateways = gateways
        self.random_starting_zone = starting_zone_polygon_points
        
        self.current_gateway_index = -1
        self.current_point = []
        self.gateway_points = []
        
        self.speeds = speeds
        self.speed = random.choice(self.speeds)
                
        self.frames_per_second = frames_per_second
       
        self.angle = 0
        self.size = 5  # Taille du triangle ajustée
        self.circle_radius = self.size + 3
        self.target_angle = 0
        
      
        self.latest_speed = self.speed
        self.latest_target_angle = 0
        
        
        self.pilot_target_angle = 0
        self.pilot_speed = 0
        
        
        self.event_active = False  # Variable pour indiquer si un événement est actif
        self.event_start_time = 0  # Variable pour enregistrer le temps de début de l'événement
        self.event_duration = 2  # Durée de l'événement
        self.conso = 100.0  # Consommation en pourcent
        
        # self.calculate_targetpoint_on_gateways()
    
    
    def get_current_point(self):
        return self.current_point[0], self.current_point[1]
        
    def set_current_point(self, x, y):
        self.current_point = (x, y)
    
    
    def random_point_in_triangle(self, triangle):
        """Generates a random point inside a triangle using barycentric coordinates."""
        r1 = random.random()
        r2 = random.random()

        # Ensure r1 + r2 < 1 for a valid point inside the triangle
        if r1 + r2 >= 1:
            r1 = 1 - r1
            r2 = 1 - r2

        a, b, c = triangle
        x = a[0] + r1 * (b[0] - a[0]) + r2 * (c[0] - a[0])
        y = a[1] + r1 * (b[1] - a[1]) + r2 * (c[1] - a[1])
        return (x, y)

    def random_point_in_quadrilateral(self, quadrilateral):
        """Generates a random point inside a quadrilateral by triangulating it."""
        # Triangulate the quadrilateral into two triangles
        tri1 = [quadrilateral[0], quadrilateral[1], quadrilateral[2]]
        tri2 = [quadrilateral[0], quadrilateral[2], quadrilateral[3]]

        # Randomly choose one of the triangles
        if random.random() < 0.5:
            return self.random_point_in_triangle(tri1)
        else:
            return self.random_point_in_triangle(tri2)

      
    
    
    def init_starting_position(self):
        # Calculer les coordonnées du point aléatoire dans la zone de départ
        random_point = self.random_point_in_quadrilateral(self.random_starting_zone)
        
        self.current_point = ( random_point[0],random_point[1])
            
       
        
    def calculate_targetpoint_on_gateways(self):
        # Sélectionner un point de passage unique sur chacun des gateways du circuit
        self.path_points = []
        for gateway in self.gateways:
            # Calculer un point de passage aléatoire sur chaque gateway
            start_point, end_point = gateway
            
            t = random.random()

            # Calculer les coordonnées du point aléatoire sur le segment
            gateway_point = (
                start_point[0] + t * (end_point[0] - start_point[0]),
                start_point[1] + t * (end_point[1] - start_point[1])
            )
            
            self.gateway_points.append(gateway_point)
            
        
        
    
    # Méthode pour gérer les événements aléatoires
    def handle_random_events(self):
        
        self.latest_speed = self.speed
        self.latest_target_angle = self.target_angle
        
        
        if not self.event_active:
            # Générer aléatoirement un événement
            if random.random() < 0.01:  # Ajustez la probabilité d'apparition de l'événement selon vos besoins
                # Sélectionner aléatoirement un événement parmi ceux disponibles
                #event_type = random.choice(["stop", "speed_up", "slow_down", "turn_right", "turn_left"])
                
                event_type = random.choice(["stop", "speed_up", "slow_down", "speed_up_with_upper_consumption",
                "slow_down_with_upper_consumption", "random_pilot_direction"])
                
                # Début de l'événement
                self.start_random_event(event_type)
                
        else:
            # Vérifier si l'événement est terminé
            if time.time() - self.event_start_time >= self.event_duration:
                # Fin de l'événement, réinitialiser le bateau à son état initial
                self.reset_random_event()

    # Méthode pour commencer un événement
    def start_random_event(self, event_type):
        self.event_active = True
        self.event_start_time = time.time()
                
        # Traiter différents types d'événements
        if event_type == "stop":
            self.speed = self.speeds[0]
        
        elif event_type == "speed_up":
        
            self.speed *= 2 
            
            if self.speed > self.speeds[len(self.speeds)-1] :
            
                self.speed = self.speeds[len(self.speeds)-1]
           
        elif event_type == "slow_down":
        
            self.speed /= 2
            
            if self.speed < self.speeds[1] :
                self.speed = self.speeds[1]
            
        elif event_type == "turn_right":
        
            self.target_angle += random.choice([-50.0, -40.0, -30.0, -20.0, -10.0])
            
        elif event_type == "turn_left":
            self.target_angle += random.choice([10.0, 20.0, 30.0, 40.0, 50.0])
            
        self.pilot_target_angle = self.target_angle
        self.pilot_speed = self.speed    
        

    # Méthode pour réinitialiser le bateau après un événement
    def reset_random_event(self):
        self.event_active = False
                
        self.speed = self.latest_speed
        self.target_angle = self.latest_target_angle
        
        
    def move(self, elapsed_time):
    
        # Gérer les événements aléatoires
        self.handle_random_events()
    
        # Déterminer le point de passage suivant en créant une boucle circulaire à travers la liste des points de passage 
        # pour permettre au véhicule de continuer son parcours autour du circuit indéfiniment.
        next_point_index = (self.current_gateway_index + 1) % len(self.gateway_points)
        next_point = self.gateway_points[next_point_index]
                  

        # Calculer la direction vers le point de passage suivant
        dx = next_point[0] - self.current_point[0]
        dy = next_point[1] - self.current_point[1]
        
        # Calculer l'angle vers le point de passage suivant
        self.target_angle = math.atan2(dy, dx)
                
        distance_to_next_point = math.sqrt(dx ** 2 + dy ** 2)
        
        scaled_speed  = self.speed * elapsed_time
        
        
        move_distance = min(scaled_speed, distance_to_next_point)
        
        # move_distance = min(scaled_speed * elapsed_time, distance_to_target)
        
                
        # Déplacer le concurrent vers le point de passage suivant      
        self.current_point = (
                
            self.current_point[0] + move_distance  * math.cos(self.target_angle),
            self.current_point[1] + move_distance * math.sin(self.target_angle)
        )

             
        # Vérifier si le concurrent a atteint le point de passage suivant
        if distance_to_next_point <= move_distance :
                                 
            previous_point_index = (self.current_gateway_index - 1) % len(self.gateway_points)
            start_point, end_point = self.gateways[previous_point_index]
           
                       
            new_previous_path_point = (
                random.randint(math.floor(min(start_point[0], end_point[0])), math.ceil(max(start_point[0], end_point[0]))),
                random.randint(math.floor(min(start_point[1], end_point[1])), math.ceil(max(start_point[1], end_point[1])))
            )
            
            
            self.gateway_points[previous_point_index] = new_previous_path_point
            
            # Mise à jour point de passage en cours
            self.current_gateway_index = next_point_index
         
            # Modifier aléatoirement la vitesse des concurrents à chaque passage
            self.speed = random.choice(self.speeds)
            

    





  

class EnergyPyBoat(PyBoat):
    def __init__(self, color, gateways, 
                    starting_zone_polygon_points,
                    speeds,
                    frames_per_second,                    
                    detection_ranges, 
                    detection_ranges_px_for_display, 
                    detector_angle_of_view, 
                    nb_detection_sectors,
                    detector_values):
        super().__init__(color, gateways, starting_zone_polygon_points, speeds, frames_per_second)
        
        self.other_red_circles = []
        
        self.detection_sector_points_per_range = []  
        self.detection_sector_points_per_range_for_display = []  
        self.detected_sectors_per_range = []
         # Initialiser une liste de listes de dimensions m x n avec des zéros
        self.init_detected_sectors_per_range = detector_values
        self.detected_sectors_per_range = self.init_detected_sectors_per_range
        
        self.detection_ranges = detection_ranges
        self.detection_ranges_for_display = detection_ranges_px_for_display
        
        self.detector_angle_of_view = detector_angle_of_view
        
        self.nb_detection_sectors = nb_detection_sectors
        
        self.sector_detection_angle = detector_angle_of_view / nb_detection_sectors
                  
        self.random_starting_zone = starting_zone_polygon_points
        
                
        self.agent_direction = 0.0
        self.agent_speed = 0.0
        
        self.pilot_target_angle = 0.0
        self.pilot_speed = 0.0
        
    def init_target_angle(self, value):
        self.pilot_target_angle = value
        self.target_angle = value  

    def set_boat_target_angle(self, value):
        self.target_angle = value  
    
    def get_boat_target_angle(self):
        return self.target_angle  

    def set_boat_speed(self, value):
        self.speed = value
        
    def get_boat_speed(self):
        return self.speed        
    
    def get_pilot_target_angle(self):
        return self.pilot_target_angle
    
    def get_pilot_speed(self):
        return self.pilot_speed  

    def get_nb_detection_sectors(self):
        return self.nb_detection_sectors
         
    
    def get_detection_sector_points(self):
        return self.detection_sector_points   

    def get_detection_sector_points_per_range(self):
        return self.detection_sector_points_per_range 

    def get_detection_sector_points_per_range_for_display(self):
        return self.detection_sector_points_per_range_for_display         
        
       
    def get_detection_ranges(self):
        return self.detection_ranges 

    def get_detected_sectors_per_range(self):
        return self.detected_sectors_per_range         
        
    def get_detections_state(self):
        state = []
        for range_of_detection_sectors in self.detected_sectors_per_range:
            for sector_detection_value in range_of_detection_sectors:
                state.append(sector_detection_value)
        
        return state
        
    def set_position(self, new_pos):
        
        # Calculer les coordonnées du point aléatoire sur le segment
        self.current_point = (new_pos[0], new_pos[1])
        
        self.moveDetectorFieldOfView()
    
    
    def init_starting_position(self):
        
        random_point = self.random_point_in_quadrilateral(self.random_starting_zone)

        # Calculer les coordonnées du point aléatoire sur le segment
        self.current_point = (random_point[0], random_point[1])
        
        self.moveDetectorFieldOfView()
        
    def init_position_on_starting_line(self, starting_line):
        
        start_point, end_point = starting_line
        
        t = random.random()

        # Calculer les coordonnées du point aléatoire sur le segment
        random_point = (
            start_point[0] + t * (end_point[0] - start_point[0]),
            start_point[1] + t * (end_point[1] - start_point[1])
        )

        self.current_point = ( random_point[0],random_point[1])
        
            
        
    def check_collision_with_sector(self, circle_center, sector_points):
        # Point au sommet du triangle avant du bateau
        boat_front_point = self.current_point
        collision = False
        
        # Trouver le point d'intersection entre la ligne reliant le centre du cercle concurrent et le sommet du triangle
        # et le segment opposé du secteur
        intersection_point = self.get_intersection_point(circle_center, boat_front_point, sector_points[1], sector_points[2])

        if intersection_point:
            # Vérifier si le point d'intersection est à l'intérieur du segment opposé
            if self.is_point_inside_segment(intersection_point, sector_points[1], sector_points[2]):
                # Calculer la distance entre le point d'intersection et le centre du cercle concurrent
                distance_to_intersection = math.dist(circle_center, intersection_point)
                
                # Vérifier s'il y a collision en comparant la distance avec le rayon du cercle concurrent
                if distance_to_intersection <= self.circle_radius:
                    collision = True
                    
        return collision
      
        
    def get_intersection_point(self, point1, point2, point3, point4):
        # Récupérer les coordonnées des points
        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = point3
        x4, y4 = point4
        
        # Calculer les coordonnées de l'intersection
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denominator == 0:
            return None  # Les lignes sont parallèles ou confondues
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return px, py

    
    def is_point_inside_segment(self, point, segment_start, segment_end):
        # Vérifier si le point est à l'intérieur du segment en vérifiant si ses coordonnées x et y sont
        # comprises entre les coordonnées x et y des points de début et de fin du segment
        x, y = point
        x1, y1 = segment_start
        x2, y2 = segment_end
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)
    
        
    def get_concurrents_position_and_check_collision(self, all_concurrents_boats):
        self.other_red_circles = []
                        
        self.moveDetectorFieldOfView()
        
        for range_num in range(len(self.detection_ranges)):
            for sector_num in range(len(self.detected_sectors_per_range[range_num])):
                self.detected_sectors_per_range[range_num][sector_num] = 0
        
        
        for boat in all_concurrents_boats:
            self.other_red_circles.append(boat.current_point)
            
                              
        if(len(self.other_red_circles)!=0 ):  
                            
            for range_num, detection_sector_points in enumerate(self.detection_sector_points_per_range):
                                
                for sector_num, sector_points in enumerate(detection_sector_points):
                    
                    for circle_center in self.other_red_circles:
                        check = self.check_collision_with_sector(circle_center, sector_points)
                        
                        if check == True:
                                                   
                            # self.detected_sectors_per_range[range_num][sector_num] = 1
                            
                            # Marquer toutes les cases dans la même colonne pour des ranges supérieurs
                            for k in range(range_num, len(self.detection_ranges)):
                                self.detected_sectors_per_range[k][sector_num] = 1    
                        
   
    def calculate_start_angles(self):
        
        
        
        if self.nb_detection_sectors % 2 == 0:
            raise ValueError("Le nombre de secteurs doit être impair.")

        # Liste des angles de départ pour chaque secteur
        start_angles = []
        
        # Indice du secteur central
        central_sector_index = self.nb_detection_sectors // 2

        for i in range(self.nb_detection_sectors):
            # Calcul de l'angle de départ pour chaque secteur
            offset = (i - central_sector_index) * self.sector_detection_angle
            angle = self.target_angle + math.radians(offset) + math.radians(self.sector_detection_angle / 2)
                    
            start_angles.append(angle)

        return start_angles
        
    
    def is_moving_forward(self, target_gateway):
        
        is_moving_forward = False
        
        boat_front_point = []
        
        # Indice du secteur central
        central_sector_index = self.nb_detection_sectors // 2
        
        sector_points = self.detection_sector_points_per_range[0][central_sector_index]
        
        sector_points[1]
        sector_points[2]
        
        
        
        sector_midpt_x, sector_midpt_y = calculate_midpoint(sector_points[1][0], 
                                                        sector_points[1][1], 
                                                        sector_points[2][0], 
                                                        sector_points[2][1])
    
        target_gateway_midpt_x, target_gateway_midpt_y = calculate_midpoint(target_gateway[0][0], 
                                                        target_gateway[0][1], 
                                                        target_gateway[1][0], 
                                                        target_gateway[1][1])
        
        sector_midpt_dist_to_gateway = calculate_distance(sector_midpt_x, 
                                                        sector_midpt_y, 
                                                        target_gateway_midpt_x, 
                                                        target_gateway_midpt_y)
                                                        
                                                        
        boat_dist_to_gateway = calculate_distance(sector_points[0][0], 
                                                        sector_points[0][1], 
                                                        target_gateway_midpt_x, 
                                                        target_gateway_midpt_y)                                                
        
        if boat_dist_to_gateway >  sector_midpt_dist_to_gateway :
            is_moving_forward = True
        
        
        return is_moving_forward
    
    def moveDetectorFieldOfView(self):    
                
        self.start_angles = self.calculate_start_angles()

        # Liste des points de chaque secteur pour un range donné 
        # on crée des coordonnées de secteurs 'detection_sector_points_per_range_for_display' 
        # pour mieux les voir à l'écran mais la détection s'applique sur les véritables portées
        self.detection_sector_points_per_range = []
        self.detection_sector_points_per_range_for_display = []
        
               
        for detection_range, detection_range_for_display in zip(self.detection_ranges, self.detection_ranges_for_display):
            detection_sector_points = []
            detection_sector_points_for_display = []
            for angle in self.start_angles:
                # Coordonnées des points pour chaque secteur
                sector_points = [
                    self.current_point,
                    (self.current_point[0] + detection_range * math.cos(angle), self.current_point[1] + detection_range * math.sin(angle)),
                    (self.current_point[0] + detection_range * math.cos(angle - math.radians(self.sector_detection_angle)), self.current_point[1] + detection_range * math.sin(angle - math.radians(self.sector_detection_angle)))
                ]
                detection_sector_points.append(sector_points)
                
                sector_points_for_display = [
                    self.current_point,
                    (self.current_point[0] + detection_range_for_display * math.cos(angle), self.current_point[1] + detection_range_for_display * math.sin(angle)),
                    (self.current_point[0] + detection_range_for_display * math.cos(angle - math.radians(self.sector_detection_angle)), self.current_point[1] + detection_range_for_display * math.sin(angle - math.radians(self.sector_detection_angle)))
                ]
                detection_sector_points_for_display.append(sector_points_for_display)
               
            self.detection_sector_points_per_range.append(detection_sector_points)
            self.detection_sector_points_per_range_for_display.append(detection_sector_points_for_display)
        
        
    # Déplacer le bateau avec les consignes de direction et de vitesse de l'agent
    def move(self, agent_direction, agent_speed, activate_pilot_behavior, elapsed_time):
        
        # print(f"agent_direction: {agent_direction}, agent_speed: {agent_speed}, activate_pilot_behavior: {activate_pilot_behavior}, elapsed_time: {elapsed_time}")
    
        self.target_angle += agent_direction
        self.speed = agent_speed 
        
        if activate_pilot_behavior :
            # Gérer les événements aléatoires correspondant 
            # à des choix délibérément contradictoires du pilote
            self.handle_random_events()
        
        else :        
            self.pilot_target_angle = self.target_angle
            self.pilot_speed = self.speed
        
        target_angle = self.pilot_target_angle
        scaled_speed = self.pilot_speed * elapsed_time
        
                       
        # Déplacer le concurrent vers le point de passage suivant      
        self.current_point = (
                
            self.current_point[0] + scaled_speed  * math.cos(target_angle),
            self.current_point[1] + scaled_speed * math.sin(target_angle)
        )
        
        
        self.moveDetectorFieldOfView()
        
        
        
        
        
        
        
        
        
    def move_to_gateways(self, elapsed_time):
        
        # Déplacement du bateau
        super().move(elapsed_time)
        
        self.pilot_target_angle = self.target_angle
        self.pilot_speed = self.speed
        
        self.moveDetectorFieldOfView()
        
        
        
# if __name__ == '__main__': 


    # directory = ''

    # track_parameters_file = '../scenario_parameters.yaml'

    # env_config={
            # "render_flag": True,
            # "activate_pilot_behavior": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'activate_pilot_behavior'),
            # "monaco_map":directory + 'assets/monaco_map_bgd.png',
            # "font":directory + 'assets/font/arial.ttf',
            # "implementation":"simple",
            # "critical_battery_level": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'critical_battery_level'),
            # "sufficient_battery_level": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'sufficient_battery_level'),
            # "minimal_required_laps": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'minimal_required_laps'),
            # "time_scaling_factor": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'time_scaling_factor'),
            # "concurrents_time_scaling_factor": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'concurrents_time_scaling_factor'),
            # "max_race_time": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_race_time'),
            # "max_loop_track_time": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_loop_track_time'),
            # "base_consumption_rate": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'base_consumption_rate'),
            # "speed_consumption_factor": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'speed_consumption_factor'),
            # "rayon_bouee": load_values_from_yaml(track_parameters_file, 'track_drawings', 'rayon_bouee'),
            # "bouees": load_bouees_from_yaml(track_parameters_file, 'track_drawings', 'bouees'),
            # "big_hexagon": load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'big_hexagon'),
            # "small_hexagon": load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'small_hexagon'),
            # "starting_zone_polygon":load_polygon_from_yaml(track_parameters_file, 'track_drawings', 'starting_polygon'),
            # "quadrilaterals":load_quadrilaterals_from_yaml(track_parameters_file, 'track_drawings', 'quadrilaterals'),
            # "eboat_random_starting_zone": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'eboat_random_starting_zone'),
            # "concurrents_random_starting_zone": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'concurrents_random_starting_zone'),
            # "random_starting_zone_two": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'random_starting_zone_two'),
            # "random_starting_zone_three": load_polygon_from_yaml(track_parameters_file, 'energy_boat_parameters', 'random_starting_zone_three'),
            # "detection_ranges": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detection_ranges'), 
            # "detection_ranges_for_display": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detection_ranges_for_display'), 
            # "detector_angle_of_view": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detector_angle_of_view'), 
            # "nb_detection_sectors": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'nb_detection_sectors'),
            # "distance_ref_px" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'distance_ref_px'),
            # "distance_ref_m" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'distance_ref_m'),
            # "detector_values" : load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'detector_values'),
            # "directions": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'directions'),
            # "speeds": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'speeds'),
            # "concurrents_speeds": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'concurrents_speeds'),        
            # "max_speed_change": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_speed_change'),
            # "max_direction_change_base": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'max_direction_change_base'),
            # "min_direction_change": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'min_direction_change'),
            # "frames_per_second": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'frames_per_second'),
            # "earth_radius": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'earth_radius'),
            # "lat_lon_point_ref_one": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'lat_lon_point_ref_one'),
            # "image_point_ref_one": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'image_point_ref_one'),
            # "lat_lon_point_ref_two": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'lat_lon_point_ref_two'),
            # "image_point_ref_two": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'image_point_ref_two'),
            # "eboat_starting_point": load_values_from_yaml(track_parameters_file, 'energy_boat_parameters', 'eboat_starting_point'),        
        # }
            
    # earth_radius = env_config["earth_radius"]
    # lat_lon_point_ref_one = env_config["lat_lon_point_ref_one"]
    # image_point_ref_one = env_config["image_point_ref_one"]
    # lat_lon_point_ref_two = env_config["lat_lon_point_ref_two"]
    # image_point_ref_two = env_config["image_point_ref_two"]
        
        # Calcul des paramètres de transformation
    # scale_x, scale_y, x_origin, y_origin, lat_origin, lon_origin = calculate_transform_params(
                                            # image_point_ref_one[0], image_point_ref_one[1],  
                                            # lat_lon_point_ref_one[0], lat_lon_point_ref_one[1], 
                                            # image_point_ref_two[0], image_point_ref_two[1],
                                            # lat_lon_point_ref_two[0], lat_lon_point_ref_two[1],
                                            # earth_radius)
                                            
    # lat, lon = image_to_latlon(x, y, scale_x, scale_y, x_origin, y_origin, lat_origin, lon_origin, earth_radius):
    
    # x, y = latlon_to_image(lat, lon, scale_x, scale_y, x_origin, y_origin, lat_origin, lon_origin, earth_radius)