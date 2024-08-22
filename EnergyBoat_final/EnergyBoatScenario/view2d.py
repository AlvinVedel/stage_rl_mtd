#!/usr/bin/env python
# coding: utf-8

import yaml
import numpy as np
import math as math
import random as random
import pygame
from PIL import Image, ImageFont, ImageDraw

from EnergyBoatScenario.utils_env import *
from EnergyBoatScenario.bridge import *

'''
In this class, write the functions useful to draw your env in 2d
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

# Définition de la taille de la fenêtre
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720


def draw_bouees(draw, bouees, rayon_bouee):
    for bouee in bouees:
        x, y = bouee
        draw.ellipse((x - rayon_bouee, y - rayon_bouee, x + rayon_bouee, y + rayon_bouee), fill="yellow", outline="yellow", width=2)

def draw_obstacles(draw, obstacles, rayon_obstacle) :
    for obstacle in obstacles:
        x, y = obstacle
        draw.ellipse((x - rayon_obstacle, y - rayon_obstacle, x + rayon_obstacle, y + rayon_obstacle), fill="red", outline="red", width=2)

def draw_current_gateway(draw, colors, gateways, rayon_gateway) :
    #print("je suis dans draw gateway")
    #print(gateways)
    for i in range(len(gateways)):
        x = gateways[i][0]
        y = gateways[i][1]
        draw.ellipse((x - rayon_gateway[i], y - rayon_gateway[i], x + rayon_gateway[i], y + rayon_gateway[i]), fill=colors[i], width=2)
   


def draw_polygon(screen, color, points, width=0):
    pygame.draw.polygon(screen, color, points, width)
            

def draw_circuit(draw, big_hexagon, small_hexagon, starting_zone_polygon):
    #print(big_hexagon, small_hexagon, starting_zone_polygon)
    bh = [(int(coord[0]), int(coord[1])) for coord in big_hexagon]
    sh = [(int(coord[0]), int(coord[1])) for coord in small_hexagon]
    sp = [(int(coord[0]), int(coord[1])) for coord in starting_zone_polygon]

    draw.polygon(bh, outline="blue", width=2)
    draw.polygon(sh, outline="green", width=2)
    draw.polygon(sp, outline="red", width=2)

def draw_recharge(draw, zone_recharge, point_recharge, rayon_recharge):
    bh = [(int(coord[0]), int(coord[1])) for coord in zone_recharge]
    x = point_recharge[0]
    y = point_recharge[1]
    draw.polygon(bh, outline="black", width=2)
    draw.ellipse((x - rayon_recharge, y - rayon_recharge, x + rayon_recharge, y + rayon_recharge), fill="black", width=2)
    


def draw_boat_circle(draw, positions, dones, colors):
    radius=5
    for i in range(len(positions)) :
        if not dones[i] :
            draw.ellipse((positions[i, 0] - radius, positions[i, 1] - radius, positions[i, 0] + radius, positions[i, 1] + radius), outline=colors[i], fill=None, width=2)
        else :
            draw.ellipse((positions[i, 0] - radius, positions[i, 1] - radius, positions[i, 0] + radius, positions[i, 1] + radius), outline='grey', fill=None, width=2)

def draw_boat_triangle(draw, positions, directions, dones, colors):  
    boat_size = 5 
    x1 = positions[:, 0] + boat_size * np.cos(directions)
    y1 = positions[:, 1] + boat_size * np.sin(directions)

    x2 = positions[:, 0] + boat_size * np.cos(directions + 2*np.pi/3)
    y2 = positions[:, 1] + boat_size * np.sin(directions + 2*np.pi/3)

    x3 = positions[:, 0] + boat_size * np.cos(directions - 2*np.pi/3)
    y3 = positions[:, 1] + boat_size * np.sin(directions - 2*np.pi/3)

    for i in range(len(positions)) :
        points = [(x1[i], y1[i]), (x2[i], y2[i]), (x3[i], y3[i])]
        if not dones[i] :
            draw.polygon(points, fill=colors[i], width=2)
        else :
            draw.polygon(points, fill='grey', width=2)
    
        



def draw_detection_field(draw, positions, directions, dones, colors):
    for i in range(len(positions)):
        if not dones[i] :
            detection_angle = 110
            detection_range = 40
            boat_orientation = np.rad2deg(directions[i])
            start_angle = boat_orientation - detection_angle / 2
            end_angle = boat_orientation + detection_angle / 2
            while start_angle < end_angle :
                actual_angle = start_angle+12
                for j in range(10, detection_range+1, 10) :
                    start_angle_rad = np.deg2rad(start_angle)
                    end_angle_rad = np.deg2rad(actual_angle)  
                    start_x = positions[i, 0] + j * math.cos(start_angle_rad)
                    start_y = positions[i, 1] + j * math.sin(start_angle_rad)
                    end_x = positions[i, 0] + j * math.cos(end_angle_rad)
                    end_y = positions[i, 1] + j * math.sin(end_angle_rad)
                    
                    # Dessiner le cône de détection autour du bateau
                    draw.polygon([(positions[i, 0], positions[i, 1]), (start_x, start_y), (end_x, end_y)], fill=None, outline=colors[i])
                start_angle+=11

    
            
            
def draw_boat_circle_info(x, y, boat, draw):
    # Dessiner le cercle rouge autour du concurrent
    draw.circle((int(x), int(y)), 4 * boat.circle_radius, width=1)

def draw_boat_triangle_info(x, y, boat, screen):   
    # Déterminer la position du triangle représentant le concurrent en utilisant les coordonnées des points de passage
    x1 = x + 3 * boat.size * math.cos(boat.target_angle)
    y1 = y + 3 * boat.size * math.sin(boat.target_angle)

    x2 = x + 3 * boat.size * math.cos(boat.target_angle + (2 * math.pi / 3))
    y2 = y + 3 * boat.size * math.sin(boat.target_angle + (2 * math.pi / 3))

    x3 = x + 3 * boat.size * math.cos(boat.target_angle - (2 * math.pi / 3))
    y3 = y + 3 * boat.size * math.sin(boat.target_angle - (2 * math.pi / 3))

    points = [(x1, y1), (x2, y2), (x3, y3)]
    pygame.draw.polygon(screen, boat.color, points)     

def draw_compass(draw, center_x, center_y, agent_direction_choice, font_path, direction_objectif):
    radius = 50
    radius_grd = 90
    radius_small = 10

    # Dessiner les cercles
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=GRAY)
    draw.ellipse((center_x - radius_grd, center_y - radius_grd, center_x + radius_grd, center_y + radius_grd), outline=GRAY, width=2)
    
    # Dessiner les graduations
    for angle in range(-180, 181, 10):
        rad = math.radians(angle - 90)  # Ajouter 90 degrés pour faire pivoter
        x = center_x + 1.5 * radius * math.cos(rad)
        y = center_y + 1.5 * radius * math.sin(rad)
        if angle % 20 == 0 and angle != 180:
            text = f"{angle}°"
            font = ImageFont.truetype(font_path, 12)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x - text_width / 2, y - text_height / 2), text, fill=BLUE, font=font)

    # Dessiner l'aiguille pour la direction de l'agent
    agent_degre = math.degrees(agent_direction_choice)
    agent_direction_rad = math.radians(agent_degre)  # Ajouter 90 degrés pour faire pivoter
    needle_x = center_x + radius * math.cos(agent_direction_rad)
    needle_y = center_y + radius * math.sin(agent_direction_rad)
    draw.line((center_x, center_y, needle_x, needle_y), fill=RED, width=2)

    # Dessiner l'aiguille pour la direction de l'objectif
    objectif_degre = math.degrees(direction_objectif)
    objectif_direction_rad = math.radians(objectif_degre)  # Ajouter 90 degrés pour faire pivoter
    needle_obj_x = center_x + radius * math.cos(objectif_direction_rad)
    needle_obj_y = center_y + radius * math.sin(objectif_direction_rad)
    draw.line((center_x, center_y, needle_obj_x, needle_obj_y), fill=GREEN, width=2)

    # Dessiner le petit cercle central
    draw.ellipse((center_x - radius_small, center_y - radius_small, center_x + radius_small, center_y + radius_small), fill=BLACK)
    


def draw_speed_dial(draw, center_x, center_y, agent_speed, font_path):
    radius = 50
    radius_grd = 90
    radius_small = 10

    # Dessiner les cercles
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), outline=GRAY)
    draw.ellipse((center_x - radius_grd, center_y - radius_grd, center_x + radius_grd, center_y + radius_grd), outline=GRAY, width=2)

    # Dessiner les graduations de 0 à 50 km/h
    for speed in range(0, 31, 1):
        angle = (speed / 30) * 180 - 180  # Convertir la vitesse en angle
        rad = math.radians(angle)
        x = center_x + 1.5 * radius * math.cos(rad)
        y = center_y + 1.5 * radius * math.sin(rad)
        if speed % 5 == 0:  # Dessiner les graduations principales toutes les 5 km/h
            text = f"{speed/2}"
            font = ImageFont.truetype(font_path, 12)
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            draw.text((x - text_width / 2, y - text_height / 2), text, fill=BLUE, font=font)

    
    # Dessiner l'aiguille pour le choix de vitesse de l'agent
    agent_speed_angle = (agent_speed / 15) * 180 - 180  # Convertir la vitesse en angle
    agent_speed_rad = math.radians(agent_speed_angle)
    needle_x = center_x + radius * math.cos(agent_speed_rad)
    needle_y = center_y + radius * math.sin(agent_speed_rad)
    draw.line((center_x, center_y, needle_x, needle_y), fill=RED, width=2)

    # Dessiner le petit cercle central
    draw.ellipse((center_x - radius_small, center_y - radius_small, center_x + radius_small, center_y + radius_small), fill=BLACK)



def draw_table(draw, table_x, table_y, table_values, font_path):
    # Dimensions du tableau
    TABLE_HEIGHT = 200
    NUM_ROWS = len(table_values)
    NUM_COLS = 2
    CELL_WIDTH = 200
    CELL_HEIGHT = TABLE_HEIGHT // NUM_ROWS

    font = ImageFont.truetype(font_path, 12)

    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            # Dessin de la cellule
            rect_x = table_x + col * CELL_WIDTH
            rect_y = table_y + row * CELL_HEIGHT
            draw.rectangle([rect_x, rect_y, rect_x + CELL_WIDTH, rect_y + CELL_HEIGHT], outline=GRAY, fill=WHITE)

            # Dessin de la valeur dans la cellule
            value_text = table_values[row][col]
            text_bbox = draw.textbbox((0, 0), value_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = rect_x + (CELL_WIDTH - text_width) / 2
            text_y = rect_y + (CELL_HEIGHT - text_height) / 2
            draw.text((text_x, text_y), value_text, fill=(0, 0, 0), font=font)



    
def draw_agent_info(draw, font_path, direction, vitesse, remaining_time, laps_completed, gateway_reached, battery_level, reward, direction_objectif,
                    boat_x, boat_y, couleur_suivi):
    
   

    info_box_width = SCREEN_WIDTH // 3
    info_box_height = SCREEN_HEIGHT // 1.8

    # Dessiner le fond blanc pour la zone d'affichage
    draw.rectangle([(SCREEN_WIDTH-info_box_width, SCREEN_HEIGHT-info_box_height), (SCREEN_WIDTH, SCREEN_HEIGHT)], fill=(255, 255, 255))

    x = info_box_width // 4
    y = 3 * (info_box_height // 4)
   
    table_values = [
        ["Remaining steps:", f"{round(remaining_time)}"],
        ["Pilot behavior:", "Autopilot"],
        ["Boat position (x,y)", f"({round(boat_x)},{round(boat_y)})"],
        ["Agent direction/speed:", f"{np.round(direction,1)}/{np.round(vitesse, 1)}"],
        ["Number of laps:", f"{laps_completed} / {gateway_reached}"],
        ["Battery level (%) / Recompense :", f"{battery_level.astype(np.int8)}/{reward.astype(np.int8)}"],    
        [ "Agent suivi :", f" Robot {couleur_suivi}"]          
    ]
    table_x = SCREEN_WIDTH - info_box_width + 10
    table_y = SCREEN_HEIGHT  - 200
    
    # Dessiner la table
    draw_table(draw, table_x, table_y, table_values, font_path)
                    
    draw_compass(draw, SCREEN_WIDTH - x, SCREEN_HEIGHT - y, direction, font_path, direction_objectif)
    
    draw_speed_dial(draw, SCREEN_WIDTH - 3 * x, SCREEN_HEIGHT - y, vitesse, font_path)


def draw_detections_table(draw, font_path, state):
    
    NUM_ROWS = 21
    NUM_COLS = 12
    CELL_WIDTH = 20
    CELL_HEIGHT = 15
            
    info_box_width = 370
    info_box_height = SCREEN_HEIGHT // 4

    x = info_box_width // 4
    y = 3 * (info_box_height // 4)

    table_values = []
    header = [" "]

    angles = np.linspace(-55, 55, 11, dtype=np.float16)
    values = np.abs(angles)
    header += [str(i)+"°" for i in values] 
    table_values.append(header)

    state = state.astype(np.int8)

    for i in range(20, 0, -1) :
        header = [str(int(i*2)) + " m"]
        idx = 200+i
        th = []
        while idx >0 :
            th.append(str(state[idx-1]))
            idx-=20
        th = th[::-1]
        header += th  
        table_values.append(header)

    table_x = 5
    table_y = 5

    font = ImageFont.truetype(font_path, 8)
    for row in range(NUM_ROWS):
        
        for col in range(NUM_COLS):
            
            rect_x = table_x + col * CELL_WIDTH
            rect_y = table_y + row * CELL_HEIGHT
            rect = [(rect_x, rect_y), (rect_x + CELL_WIDTH, rect_y + CELL_HEIGHT)]
                
            if row > 0 and table_values[row][col] == "2":
                draw.rectangle(rect, fill=(255, 0, 0))  # Rouge pour "1"
            elif row > 0 and table_values[row][col] == "0":
                draw.rectangle(rect, fill=(0, 255, 0))  # Vert pour "0"
            elif row > 0 and table_values[row][col] == "1":
                draw.rectangle(rect, fill=(0, 0, 255))  # Bleu pour "1"
            else:
                draw.rectangle(rect, fill=(255, 255, 255))  # Blanc pour l'entête

                # Dessin de la valeur dans la cellule
            if row==0  or col==0:
                value_text = table_values[row][col]
                text_bbox = draw.textbbox((0, 0), value_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]           
                text_x = rect_x + (CELL_WIDTH - text_width) / 2
                text_y = rect_y + (CELL_HEIGHT - text_height) / 2
                draw.text((text_x, text_y), value_text, fill=(0, 0, 0), font=font)

                # Dessin du contour de la cellule
            draw.rectangle(rect, outline=(128, 128, 128), width=1)