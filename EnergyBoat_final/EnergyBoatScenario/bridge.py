import numpy as np
import math
import random
import socket
import json
import time

from EnergyBoatScenario.utils_env import *

class MobileCommonInterface:
    
    def move_to(self, coordinates):
        raise NotImplementedError

    def get_position(self):
        raise NotImplementedError

    def get_battery_level(self):
        raise NotImplementedError
     
        
        
    def set_battery_level(self, value):
        raise NotImplementedError

    def reset_battery_level(self):
        raise NotImplementedError
            
    def init_starting_position(self):
        raise NotImplementedError
        
    def init_position_on_starting_line(self, starting_line):
        raise NotImplementedError      
        
    def get_current_pos(self):
        raise NotImplementedError
        
    def set_position(self, new_pose):
        raise NotImplementedError
        
    def get_detections_state(self):
        raise NotImplementedError 
        
    def get_detected_sectors_per_range(self):
        raise NotImplementedError 

    def get_boat_object(self):
        raise NotImplementedError

    def set_boat_object(self, value):
        raise NotImplementedError
        
    def set_agent_direction_choice(self, agent_direction_choice):
        raise NotImplementedError
        
    def set_agent_speed_choice(self, agent_speed_choice):
        raise NotImplementedError   
    
    
    def get_agent_direction_choice(self):
        raise NotImplementedError
        
    def get_agent_speed_choice(self):
        raise NotImplementedError 

    def init_target_angle(self, value): 
        raise NotImplementedError
        
    def set_target_angle(self, value):
        raise NotImplementedError
        
    def get_pilot_target_angle(self):
        raise NotImplementedError   
    
    def get_pilot_speed(self):    
        raise NotImplementedError  

    def get_boat_target_angle(self):
        raise NotImplementedError  

    def set_boat_speed(self, value):
        raise NotImplementedError 
        
    def get_boat_speed(self):
        raise NotImplementedError    

    def move(self, agent_direction, agent_speed, activate_pilot_behavior, elapsed_time):
        raise NotImplementedError         
    
class SimImplementation(MobileCommonInterface):
    
    def __init__(self, energyPyBoat):
        self.x = 0
        self.y = 0
        self.pyBoat = energyPyBoat
        self.agent_direction_choice = 0
        self.agent_speed_choice = 0
        self.battery_level = 100.0
    
    def get_current_pos(self):
        self.x, self.y = self.get_pyboat_object().get_current_point()
        return self.x, self.y
   
    def set_position(self, new_pose):
        self.x = new_pose[0]
        self.y = new_pose[1]
        self.get_pyboat_object().set_position(new_pose)

    def get_info(self):  
        return "I'm an instance of SimImplementation"

    def get_pyboat_object(self):
        return self.pyBoat

    def set_pyboat_object(self, value):
        self.pyBoat = value
        
    def get_battery_level(self):
        return self.battery_level
        
    def set_battery_level(self, value):
        self.battery_level = value  
    
    def init_starting_position(self):
        self.get_pyboat_object().init_starting_position() 
        
    def init_position_on_starting_line(self, starting_line):
        self.get_pyboat_object().init_position_on_starting_line(starting_line)    
        
    def init_target_angle(self, value):    
        self.get_pyboat_object().init_target_angle(value)
        
    def set_target_angle(self, value):
        self.get_pyboat_object().set_boat_target_angle(value)  

    def get_detections_state(self):
        return self.get_pyboat_object().get_detections_state() 
        
    def get_detected_sectors_per_range(self):
        return self.get_pyboat_object().get_detected_sectors_per_range()  
        
    def set_agent_direction_choice(self, agent_direction_choice):
        self.agent_direction_choice = agent_direction_choice
        
    def set_agent_speed_choice(self, agent_speed_choice):
        self.agent_speed_choice = agent_speed_choice 


    def get_agent_direction_choice(self):
        return self.agent_direction_choice
        
    def get_agent_speed_choice(self):
        return self.agent_speed_choice   
         
    def get_pilot_target_angle(self):
        return self.get_pyboat_object().get_pilot_target_angle()  
    
    def get_pilot_speed(self):    
        return self.get_pyboat_object().get_pilot_speed() 
     
    def get_boat_target_angle(self):
        return self.get_pyboat_object().get_boat_target_angle()

    def set_boat_speed(self, value):
        return self.get_pyboat_object().set_boat_speed(value)

    def get_boat_speed(self):
        return self.get_pyboat_object().get_boat_speed()        
    
    def move(self, agent_direction, agent_speed, activate_pilot_behavior, elapsed_time):
        self.get_pyboat_object().move(agent_direction, agent_speed, activate_pilot_behavior, elapsed_time)

class RealImplementation(MobileCommonInterface):
    
    def __init__(self):
        self.x = 0
        self.y = 0
        self.agent_direction_choice = 0
        self.agent_speed_choice = 0
        self.battery_level = 100.0  

    def get_battery_level(self):
        return self.battery_level
        
    def set_battery_level(self, value):
        self.battery_level = value 
    
    
    def set_position(self, new_pose):
        raise NotImplementedError
        
    def init_target_angle(self, value): 
        raise NotImplementedError
        
    def set_target_angle(self, value):
        raise NotImplementedError

    def get_info(self):  
        return "I'm an instance of RealImplementation"

    def get_pyboat_object(self):
        raise NotImplementedError

    def set_pyboat_object(self, value):
        raise NotImplementedError
        
    def init_starting_position(self):
        raise NotImplementedError
        
    def init_position_on_starting_line(self, starting_line):
        raise NotImplementedError   
        
    def get_current_pos(self):
        raise NotImplementedError
        
    def get_detections_state(self):
        raise NotImplementedError 

    def get_detected_sectors_per_range(self):
        raise NotImplementedError

    def set_agent_direction_choice(self, agent_direction_choice):
        self.agent_direction_choice = agent_direction_choice
        
    def set_agent_speed_choice(self, agent_speed_choice):
        self.agent_speed_choice = agent_speed_choice 

    def get_agent_direction_choice(self):
        return self.agent_direction_choice
        
    def get_agent_speed_choice(self):
        return self.agent_speed_choice
        
        
    def get_pilot_target_angle(self):
        raise NotImplementedError   
    
    def get_pilot_speed(self):    
        raise NotImplementedError 

    def get_boat_target_angle(self):
        raise NotImplementedError
        
    def set_boat_speed(self, value):
        raise NotImplementedError 
        
    def get_boat_speed(self):
        raise NotImplementedError 
        

    def move(self, agent_direction, agent_speed, activate_pilot_behavior, elapsed_time):
        raise NotImplementedError        

class EnergyBoat:

    def __init__(self, implementation_type="simple", *args, **kwargs):
        if implementation_type == "simple":
            print("I'm sim !")
            self.implementation = SimImplementation(*args, **kwargs)
        elif implementation_type == "real":
            print("I'm real !")
            self.implementation = RealImplementation(*args, **kwargs)
        else: 
            raise ValueError("Incorrect implementation value. Choose 'real' or 'simple'.")

    def get_battery_level(self):
        return self.implementation.get_battery_level()
        
    def set_battery_level(self, value):
        return self.implementation.set_battery_level(value)
   
    def get_pos(self):
        return self.implementation.get_pos()
     
    def set_position(self, new_pose):
        return self.implementation.set_position(new_pose)
   
    def init_target_angle(self, value): 
        return self.implementation.init_target_angle(value)
        
    def set_target_angle(self, value):
        return self.implementation.init_target_angle(value)
    
    def get_info(self): 
        return self.implementation.get_info()

    def get_pyboat_object(self):
        return self.implementation.get_pyboat_object()

    def set_pyboat_object(self, value):
        return self.implementation.set_pyboat_object(value)
        
    def init_starting_position(self):
        return self.implementation.init_starting_position()
        
    def init_position_on_starting_line(self, starting_line):
        return self.implementation.init_position_on_starting_line(starting_line)    
     
    def get_current_pos(self):
        return self.implementation.get_current_pos()
        
    def get_detections_state(self):
        return self.implementation.get_detections_state() 
        
    def get_detected_sectors_per_range(self):
        return self.implementation.get_detected_sectors_per_range() 
       
    def set_agent_direction_choice(self, agent_direction_choice):
        return self.implementation.set_agent_direction_choice(agent_direction_choice)
        
    def set_agent_speed_choice(self, agent_speed_choice):
        return self.implementation.set_agent_speed_choice(agent_speed_choice)
       
    def get_agent_direction_choice(self):
        return self.implementation.get_agent_direction_choice()
        
    def get_agent_speed_choice(self):
        return self.implementation.get_agent_speed_choice() 
            
    def get_pilot_target_angle(self):
        return self.implementation.get_pilot_target_angle()   
    
    def get_pilot_speed(self):    
        return self.implementation.get_pilot_speed()     
        
    def get_boat_target_angle(self):
        return self.implementation.get_boat_target_angle() 
        
    def set_boat_speed(self, value):
        return self.implementation.set_boat_speed(value)
        
    def get_boat_speed(self):
        return self.implementation.get_boat_speed() 

    def move(self, agent_direction, agent_speed, activate_pilot_behavior, elapsed_time):
        return self.implementation.move(agent_direction, agent_speed, activate_pilot_behavior, elapsed_time)       