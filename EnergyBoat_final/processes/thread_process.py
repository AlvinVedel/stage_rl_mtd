from processes.worker_process import WorkerProcess
from ReplayBuffer import ReplayBuffer
from RecordingBuffer import RecordingBuffer
from utils.quantile_loss import QuantileHuberLoss
import numpy as np
import tensorflow as tf
import threading
import random
import multiprocessing as mp
from EnergyBoatScenario.utils_env import *
from VAE.VAE import CVAE




class ThreadProcess(threading.Thread) :
    def __init__(self, thread_id, model, gpu_lock, names, nb_workers=4, nb_envs=4, nb_agents=4) :
        super().__init__()
        self.thread_id = thread_id
        self.gpu_lock = gpu_lock
        self.names = names

        self.nb_workers = nb_workers
        self.nb_envs = nb_envs
        self.nb_agents = nb_agents

        self.buffer = ReplayBuffer()
        self.recorder = RecordingBuffer(names=self.names)
        self.frames_count = 0
        self.episodes_count = 0

        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_factor = 0.99
        self.gamma = 0.9
        self.batch_size = 512
        self.num_actions = 25
        self.loss_function = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        
        self.pipes = []
        self.workers = []
        
        with self.gpu_lock :
            self.model_main = model
            self.model_target = model

        self.directory = '../EnergyBoatScenario/'
        self.track_parameters_file = './scenario_parameters.yaml'
        self.env_config ={
            "render_flag": True,
            "activate_pilot_behavior": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'activate_pilot_behavior'),
            "monaco_map":self.directory + 'assets/monaco_map_bgd.png',
            "font":self.directory + 'assets/font/arial.ttf',
            "implementation":"simple",
            "critical_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'critical_battery_level'),
            "sufficient_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'sufficient_battery_level'),
            "minimal_required_laps": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'minimal_required_laps'),
            "time_scaling_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'time_scaling_factor'),
            "max_race_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_race_time'),
            "max_loop_track_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_loop_track_time'),
            "base_consumption_rate": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'base_consumption_rate'),
            "speed_consumption_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speed_consumption_factor'),
            "rayon_bouee": load_values_from_yaml(self.track_parameters_file , 'track_drawings', 'rayon_bouee'),
            "bouees": load_bouees_from_yaml(self.track_parameters_file , 'track_drawings', 'bouees'),
            "big_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'big_hexagon'),
            "small_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'small_hexagon'),
            "starting_zone_polygon":load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'starting_polygon'),
            "quadrilaterals":load_quadrilaterals_from_yaml(self.track_parameters_file , 'track_drawings', 'quadrilaterals'),
            "random_starting_zone_one": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_one'),
            "random_starting_zone_two": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_two'),
            "random_starting_zone_three": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_three'),
            "detection_ranges": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges'), 
            "detection_ranges_for_display": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges_for_display'), 
            "detector_angle_of_view": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_angle_of_view'), 
            "nb_detection_sectors": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'nb_detection_sectors'),
            "distance_ref_px" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_px'),
            "distance_ref_m" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_m'),
            "detector_values" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_values'),
            "directions": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'directions'),
            "speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speeds'),
            "concurrents_speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'concurrents_speeds'),        
            "frames_per_second": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'frames_per_second'),
            "earth_radius": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'earth_radius'),
            "lat_lon_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_one'),
            "image_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_one'),
            "lat_lon_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_two'),
            "image_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_two'),
        }
        self.actionSet = {
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

        start_seed = 64
        random.seed(start_seed)
        self.workers_seed = [random.randint(0, 2**32 - 1) for _ in range(self.nb_workers)]
        for worker_id in range(self.nb_workers):
            parent_end, child_end = mp.Pipe()
            process = WorkerProcess(worker_id, child_end, self.workers_seed[worker_id], nb_envs=self.nb_envs, nb_agents=self.nb_agents, env_config=self.env_config)
            self.pipes.append(parent_end)
            self.workers.append(process)
            process.start()



        
    def run(self) :
        states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
        states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
        for pipe in self.pipes :
            pipe.send("RESET")
            states = pipe.recv()  
            states1[states[0]] = states[1]  
            states2[states[0]] = states[2]
        states1 = states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))  
        states2 = states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))        

        # Main Loop
        while True : 

            # SAMPLE ACTION
            alea = np.random.random(size=(states1.shape[0]))
            state_tensor1 = tf.convert_to_tensor(states1) 
            state_tensor2 = tf.convert_to_tensor(states2)  
            with self.gpu_lock :
                q_vals = self.model_main([state_tensor1, state_tensor2], training=False) 
            action = tf.argmax(q_vals, axis=1).numpy() 
            actions = np.zeros((states1.shape[0], 2))
            for a in range(states1.shape[0]) :
                if alea[a] < self.epsilon :   
                    action[a] = random.randint(0, 24)
                actions[a] = self.actionSet[action[a]]
            actions = actions.reshape((self.nb_workers, self.nb_envs, self.nb_agents, 2)) 
            for i, pipe in enumerate(self.pipes) :
                pipe.send(actions[i])
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_factor)



            # GET NEXT STATE
            next_states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
            next_states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
            rl = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 3))
            steps = np.zeros((self.nb_workers, self.nb_envs))

            for i, pipe in enumerate(self.pipes) :
                obs = pipe.recv() 
                next_states1[obs[0]] = obs[1]
                next_states2[obs[0]] = obs[2]
                rl[obs[0]] = obs[3]
                steps[obs[0]] = obs[4]
            next_states1 = next_states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))
            next_states2 = next_states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))

            all_dones = rl[:, :, :, 1] 
            all_dones = np.all(all_dones, axis=2)
            rl = rl.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 3)) 


            # STORE
            self.buffer.store(states1, states2, action, rl[:, 0], rl[:, 1], next_states1, next_states2)

            
            # CHECK FOR TERMINAL STATE
            for i in range(self.nb_workers) :
                for j in range(self.nb_envs) :
                    if all_dones[i, j] or steps[i, j] >= 1200:
                        self.pipes[i].send(("RESET", j))
                        init_state, env_infos = self.pipes[i].recv()
                        print("EPISODE TERMINE :", env_infos)
                        print("NB EP: ", self.episodes_count, "frames :", self.frames_count, " - ", i, j)
                        next_states1[i*j:i*j+self.nb_agents] = init_state[0]
                        next_states2[i*j:i*j+self.nb_agents] = init_state[1]

                        self.recorder.record(env_infos[0], env_infos[1], env_infos[2], env_infos[3])
                        self.episodes_count+=1
                        
            # STATES = NEXT STATE
            states1 = next_states1
            states2 = next_states2

            # EXPERIENCE REPLAY
            if (self.buffer.memory_count > self.batch_size) :
                    state_samplev, state2_samplev, aa, rr, dones, state_next_samplev, state2_next_samplev = self.buffer.sample(self.batch_size)

                    with self.gpu_lock :
                        future_reward = self.model_target([state_next_samplev,state2_next_samplev], training=False)
                        updated_q_values = np.array(rr) +( self.gamma *  tf.reduce_max(future_reward, axis=1) )*(1-dones)

                        masks = tf.one_hot(np.array(aa), self.num_actions)
                        
                        with tf.GradientTape() as tape:
                            q_values = self.model_main([state_samplev, state2_samplev])
                            q_action = tf.reduce_sum(tf.multiply(q_values,masks), axis=1)
                            loss = self.loss_function(updated_q_values, q_action)
                        grads = tape.gradient(loss, self.model_main.trainable_variables)#model
                        self.optimizer.apply_gradients(zip(grads, self.model_main.trainable_variables))#model
                

            # SAVE WEIGHTS
            self.frames_count+=1
            if self.frames_count % 1000 == 0 :
                with self.gpu_lock :
                    self.model_target.set_weights(self.model_main.get_weights())
                    self.model_target.save(self.names["save_model"])

            # END TRAINING
            if self.episodes_count > 20000:
                for pipe in self.pipes :
                    pipe.send("EXIT")
                break





class DistributionalThreadProcess(threading.Thread) :
    """
    Meme fonctionnement que Thread classique bien que différence dans la loss : Quantile Huber Loss
    Ajout d'une opération pour le choix d'action et du paramètre de nombre de quantiles
    
    """
    def __init__(self, thread_id, model, gpu_lock, names, nb_workers=4, nb_envs=4, nb_agents=4, nb_quantiles=51) :
        super().__init__()
        self.thread_id = thread_id
        self.gpu_lock = gpu_lock
        self.names = names

        self.nb_workers = nb_workers
        self.nb_envs = nb_envs
        self.nb_agents = nb_agents

        self.buffer = ReplayBuffer()
        self.recorder = RecordingBuffer(names=names)
        self.frames_count = 0
        self.episodes_count = 0

        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_factor = 0.99
        self.gamma = 0.9
        self.batch_size = 512
        self.num_actions = 25
        self.nb_quantiles = nb_quantiles
        self.taus = np.linspace(0, 1, self.nb_quantiles+2)[1:-1]
        self.loss_function = QuantileHuberLoss(self.taus)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        
        self.pipes = []
        self.workers = []
        
        with self.gpu_lock :
            self.model_main = model
            self.model_target = model

        self.directory = '../EnergyBoatScenario/'
        self.track_parameters_file = './scenario_parameters.yaml'
        self.env_config ={
            "render_flag": True,
            "activate_pilot_behavior": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'activate_pilot_behavior'),
            "monaco_map":self.directory + 'assets/monaco_map_bgd.png',
            "font":self.directory + 'assets/font/arial.ttf',
            "implementation":"simple",
            "critical_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'critical_battery_level'),
            "sufficient_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'sufficient_battery_level'),
            "minimal_required_laps": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'minimal_required_laps'),
            "time_scaling_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'time_scaling_factor'),
            "max_race_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_race_time'),
            "max_loop_track_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_loop_track_time'),
            "base_consumption_rate": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'base_consumption_rate'),
            "speed_consumption_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speed_consumption_factor'),
            "rayon_bouee": load_values_from_yaml(self.track_parameters_file , 'track_drawings', 'rayon_bouee'),
            "bouees": load_bouees_from_yaml(self.track_parameters_file , 'track_drawings', 'bouees'),
            "big_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'big_hexagon'),
            "small_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'small_hexagon'),
            "starting_zone_polygon":load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'starting_polygon'),
            "quadrilaterals":load_quadrilaterals_from_yaml(self.track_parameters_file , 'track_drawings', 'quadrilaterals'),
            "random_starting_zone_one": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_one'),
            "random_starting_zone_two": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_two'),
            "random_starting_zone_three": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_three'),
            "detection_ranges": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges'), 
            "detection_ranges_for_display": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges_for_display'), 
            "detector_angle_of_view": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_angle_of_view'), 
            "nb_detection_sectors": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'nb_detection_sectors'),
            "distance_ref_px" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_px'),
            "distance_ref_m" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_m'),
            "detector_values" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_values'),
            "directions": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'directions'),
            "speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speeds'),
            "concurrents_speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'concurrents_speeds'),        
            "frames_per_second": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'frames_per_second'),
            "earth_radius": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'earth_radius'),
            "lat_lon_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_one'),
            "image_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_one'),
            "lat_lon_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_two'),
            "image_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_two'),
        }
        self.actionSet = {
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

        start_seed = 64
        random.seed(start_seed)
        self.workers_seed = [random.randint(0, 2**32 - 1) for _ in range(self.nb_workers)]
        for worker_id in range(self.nb_workers):
            parent_end, child_end = mp.Pipe()
            process = WorkerProcess(worker_id, child_end, self.workers_seed[worker_id], nb_envs=self.nb_envs, nb_agents=self.nb_agents, env_config=self.env_config)
            self.pipes.append(parent_end)
            self.workers.append(process)
            process.start()




        
    def run(self) :
        states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
        states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
        for pipe in self.pipes :
            pipe.send("RESET")
            states = pipe.recv()  
            states1[states[0]] = states[1]  
            states2[states[0]] = states[2]
        states1 = states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))  
        states2 = states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))        

        # Main Loop
        while True : 

            # SAMPLE ACTION
            alea = np.random.random(size=(states1.shape[0]))
            state_tensor1 = tf.convert_to_tensor(states1) 
            state_tensor2 = tf.convert_to_tensor(states2)  
            with self.gpu_lock :
                z_values = self.model_main([state_tensor1, state_tensor2], training=False) 
            mus = tf.reduce_mean(z_values, axis=2)
            action = tf.argmax(mus, axis=1).numpy() 
            actions = np.zeros((states1.shape[0], 2))
            for a in range(states1.shape[0]) :
                if alea[a] < self.epsilon :   
                    action[a] = random.randint(0, 24)
                actions[a] = self.actionSet[action[a]]
            actions = actions.reshape((self.nb_workers, self.nb_envs, self.nb_agents, 2)) 
            for i, pipe in enumerate(self.pipes) :
                pipe.send(actions[i])
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_factor)



            # GET NEXT STATE
            next_states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
            next_states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
            rl = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 3))
            steps = np.zeros((self.nb_workers, self.nb_envs))

            for i, pipe in enumerate(self.pipes) :
                obs = pipe.recv() 
                next_states1[obs[0]] = obs[1]
                next_states2[obs[0]] = obs[2]
                rl[obs[0]] = obs[3]
                steps[obs[0]] = obs[4]
            next_states1 = next_states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))
            next_states2 = next_states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))

            all_dones = rl[:, :, :, 1] 
            all_dones = np.all(all_dones, axis=2)
            rl = rl.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 3)) 


            # STORE
            self.buffer.store(states1, states2, action, rl[:, 0], rl[:, 1], next_states1, next_states2)

            
            # CHECK FOR TERMINAL STATE
            for i in range(self.nb_workers) :
                for j in range(self.nb_envs) :
                    if all_dones[i, j] or steps[i, j] >= 1200:
                        self.pipes[i].send(("RESET", j))
                        init_state, env_infos = self.pipes[i].recv()
                        print("EPISODE TERMINE :", env_infos)
                        print("NB EP: ", self.episodes_count, "frames :", self.frames_count, " - ", i, j)
                        next_states1[i*j:i*j+self.nb_agents] = init_state[0]
                        next_states2[i*j:i*j+self.nb_agents] = init_state[1]

                        self.recorder.record(env_infos[0], env_infos[1], env_infos[2], env_infos[3])
                        self.episodes_count+=1
                        
            # STATES = NEXT STATE
            states1 = next_states1
            states2 = next_states2

            # EXPERIENCE REPLAY
            if (self.buffer.memory_count > self.batch_size) :
                    state_samplev, state2_samplev, aa, rr, dones, state_next_samplev, state2_next_samplev = self.buffer.sample(self.batch_size)

                    with self.gpu_lock :
                        future_reward = self.model_target([state_next_samplev,state2_next_samplev], training=False)  # batch, 25, quantiles
                        mus = tf.reduce_mean(future_reward, axis=2)
                        a_star = tf.argmax(mus, axis=1)
                        a_star_ind = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.cast(a_star, dtype=tf.int32)], axis=1) # 128, 2
                        distr_star = tf.gather_nd(future_reward, a_star_ind)
                        updated_z_distr = np.expand_dims(np.array(rr), axis=1) + (self.gamma * distr_star)*(1 - np.expand_dims(dones, axis=1))   # 128,1 + 128, 10 * 128, 1

                        masks = tf.one_hot(np.array(aa), self.num_actions)
                        with tf.GradientTape() as tape :
                            z_distrs = self.model_main([state_samplev, state2_samplev])
                            z_distr = tf.reduce_sum(tf.multiply(z_distrs, tf.expand_dims(masks, axis=2)), axis=1)
                            loss = self.loss_function(updated_z_distr, z_distr)
                        grads = tape.gradient(loss, self.model_main.trainable_variables)#model
                        self.optimizer.apply_gradients(zip(grads, self.model_main.trainable_variables))#model
                    

            # SAVE WEIGHTS
            self.frames_count+=1
            if self.frames_count % 1000 == 0 :
                with self.gpu_lock :
                    self.model_target.set_weights(self.model_main.get_weights())
                    self.model_target.save(self.names["save_model"])

            # END TRAINING
            if self.episodes_count > 20000:
                for pipe in self.pipes :
                    pipe.send("EXIT")
                break


        



class VariationalThreadProcess(threading.Thread) :
    """
    Presque identique à script classique mais on utilise un VAE pré entrainé sur les observations conique de l'agent
    => différence dans choix de l'action 
    """
    def __init__(self, thread_id, model, vae_dim, vae_encoder, vae_decoder, gpu_lock, names, nb_workers=4, nb_envs=4, nb_agents=4) :
        super().__init__()
        self.thread_id = thread_id
        self.gpu_lock = gpu_lock
        self.names = names

        self.nb_workers = nb_workers
        self.nb_envs = nb_envs
        self.nb_agents = nb_agents

        self.buffer = ReplayBuffer()
        self.recorder = RecordingBuffer(names)
        self.frames_count = 0
        self.episodes_count = 0

        self.epsilon = 1
        self.epsilon_min = 0.1
        self.epsilon_factor = 0.99
        self.gamma = 0.9
        self.batch_size = 512
        self.num_actions = 25
        self.loss_function = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
        
        self.pipes = []
        self.workers = []

        self.vae_dim = vae_dim
        
        with self.gpu_lock :
            self.vae = CVAE(self.vae_dim)

            encoder = tf.keras.models.load_model(vae_encoder)
            new_input = tf.keras.Input(shape=(220, ))
            one_hot_encoded = tf.one_hot(tf.cast(new_input, tf.int32), depth=3)
            encoded_output = encoder(one_hot_encoded)
            self.vae.encoder = tf.keras.Model(inputs=new_input, outputs=encoded_output)
            self.vae.decoder = tf.keras.models.load_model(vae_decoder) 

            self.model_main = model
            self.model_target = model

        self.directory = '../EnergyBoatScenario/'
        self.track_parameters_file = './scenario_parameters.yaml'
        self.env_config ={
            "render_flag": True,
            "activate_pilot_behavior": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'activate_pilot_behavior'),
            "monaco_map":self.directory + 'assets/monaco_map_bgd.png',
            "font":self.directory + 'assets/font/arial.ttf',
            "implementation":"simple",
            "critical_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'critical_battery_level'),
            "sufficient_battery_level": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'sufficient_battery_level'),
            "minimal_required_laps": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'minimal_required_laps'),
            "time_scaling_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'time_scaling_factor'),
            "max_race_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_race_time'),
            "max_loop_track_time": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'max_loop_track_time'),
            "base_consumption_rate": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'base_consumption_rate'),
            "speed_consumption_factor": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speed_consumption_factor'),
            "rayon_bouee": load_values_from_yaml(self.track_parameters_file , 'track_drawings', 'rayon_bouee'),
            "bouees": load_bouees_from_yaml(self.track_parameters_file , 'track_drawings', 'bouees'),
            "big_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'big_hexagon'),
            "small_hexagon": load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'small_hexagon'),
            "starting_zone_polygon":load_polygon_from_yaml(self.track_parameters_file , 'track_drawings', 'starting_polygon'),
            "quadrilaterals":load_quadrilaterals_from_yaml(self.track_parameters_file , 'track_drawings', 'quadrilaterals'),
            "random_starting_zone_one": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_one'),
            "random_starting_zone_two": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_two'),
            "random_starting_zone_three": load_polygon_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'random_starting_zone_three'),
            "detection_ranges": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges'), 
            "detection_ranges_for_display": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detection_ranges_for_display'), 
            "detector_angle_of_view": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_angle_of_view'), 
            "nb_detection_sectors": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'nb_detection_sectors'),
            "distance_ref_px" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_px'),
            "distance_ref_m" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'distance_ref_m'),
            "detector_values" : load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'detector_values'),
            "directions": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'directions'),
            "speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'speeds'),
            "concurrents_speeds": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'concurrents_speeds'),        
            "frames_per_second": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'frames_per_second'),
            "earth_radius": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'earth_radius'),
            "lat_lon_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_one'),
            "image_point_ref_one": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_one'),
            "lat_lon_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'lat_lon_point_ref_two'),
            "image_point_ref_two": load_values_from_yaml(self.track_parameters_file , 'energy_boat_parameters', 'image_point_ref_two'),
        }
        self.actionSet = {
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

        start_seed = 64
        random.seed(start_seed)
        self.workers_seed = [random.randint(0, 2**32 - 1) for _ in range(self.nb_workers)]
        for worker_id in range(self.nb_workers):
            parent_end, child_end = mp.Pipe()
            process = WorkerProcess(worker_id, child_end, self.workers_seed[worker_id], nb_envs=self.nb_envs, nb_agents=self.nb_agents, env_config=self.env_config)
            self.pipes.append(parent_end)
            self.workers.append(process)
            process.start()



        
    def run(self) :
        states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
        states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
        for pipe in self.pipes :
            pipe.send("RESET")
            states = pipe.recv()  
            states1[states[0]] = states[1]  
            states2[states[0]] = states[2]
        states1 = states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))  
        states2 = states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))        

        # Main Loop
        while True : 

            # SAMPLE ACTION
            alea = np.random.random(size=(states1.shape[0]))
            state_tensor1 = tf.convert_to_tensor(states1) 
            state_tensor2 = tf.convert_to_tensor(states2)  
            with self.gpu_lock :
                mean, logvar = self.vae.encode(state_tensor1)
                z = self.vae.reparameterize(mean, logvar)
                x_logit = tf.argmax(self.vae.decode(z), axis=2)
                q_vals = self.model_main([x_logit, state_tensor2], training=False) 
            action = tf.argmax(q_vals, axis=1).numpy() 
            actions = np.zeros((states1.shape[0], 2))
            for a in range(states1.shape[0]) :
                if alea[a] < self.epsilon :   
                    action[a] = random.randint(0, 24)
                actions[a] = self.actionSet[action[a]]
            actions = actions.reshape((self.nb_workers, self.nb_envs, self.nb_agents, 2)) 
            for i, pipe in enumerate(self.pipes) :
                pipe.send(actions[i])
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_factor)



            # GET NEXT STATE
            next_states1 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 220))
            next_states2 = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 6))
            rl = np.zeros((self.nb_workers, self.nb_envs, self.nb_agents, 3))
            steps = np.zeros((self.nb_workers, self.nb_envs))

            for i, pipe in enumerate(self.pipes) :
                obs = pipe.recv() 
                next_states1[obs[0]] = obs[1]
                next_states2[obs[0]] = obs[2]
                rl[obs[0]] = obs[3]
                steps[obs[0]] = obs[4]
            next_states1 = next_states1.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 220))
            next_states2 = next_states2.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 6))

            all_dones = rl[:, :, :, 1] 
            all_dones = np.all(all_dones, axis=2)
            rl = rl.reshape((self.nb_workers*self.nb_envs*self.nb_agents, 3)) 


            # STORE
            self.buffer.store(states1, states2, action, rl[:, 0], rl[:, 1], next_states1, next_states2)

            
            # CHECK FOR TERMINAL STATE
            for i in range(self.nb_workers) :
                for j in range(self.nb_envs) :
                    if all_dones[i, j] or steps[i, j] >= 1200:
                        self.pipes[i].send(("RESET", j))
                        init_state, env_infos = self.pipes[i].recv()
                        print("EPISODE TERMINE :", env_infos)
                        print("NB EP: ", self.episodes_count, "frames :", self.frames_count, " - ", i, j)
                        next_states1[i*j:i*j+self.nb_agents] = init_state[0]
                        next_states2[i*j:i*j+self.nb_agents] = init_state[1]

                        self.recorder.record(env_infos[0], env_infos[1], env_infos[2], env_infos[3])
                        self.episodes_count+=1
                        
            # STATES = NEXT STATE
            states1 = next_states1
            states2 = next_states2

            # EXPERIENCE REPLAY
            if (self.buffer.memory_count > self.batch_size) :
                    state_samplev, state2_samplev, aa, rr, dones, state_next_samplev, state2_next_samplev = self.buffer.sample(self.batch_size)

                    with self.gpu_lock :
                        mean, logvar = self.vae.encode(state_next_samplev)
                        z = self.vae.reparameterize(mean, logvar)
                        x_logit_next = tf.argmax(self.vae.decode(z), axis=2)
                        future_reward = self.model_target([x_logit_next,state2_next_samplev], training=False)
                        updated_q_values = np.array(rr) +( self.gamma *  tf.reduce_max(future_reward, axis=1) )*(1-dones)

                        masks = tf.one_hot(np.array(aa), self.num_actions)
                        
                        with tf.GradientTape() as tape:
                            mean, logvar = self.vae.encode(state_samplev)
                            z = self.vae.reparameterize(mean, logvar)
                            x_logit_prev = tf.argmax(self.vae.decode(z), axis=2)
                            q_values = self.model_main([x_logit_prev, state2_samplev])
                            q_action = tf.reduce_sum(tf.multiply(q_values,masks), axis=1)
                            loss = self.loss_function(updated_q_values, q_action)
                        grads = tape.gradient(loss, self.model_main.trainable_variables)#model
                        self.optimizer.apply_gradients(zip(grads, self.model_main.trainable_variables))#model
                

            # SAVE WEIGHTS
            self.frames_count+=1
            if self.frames_count % 1000 == 0 :
                with self.gpu_lock :
                    self.model_target.set_weights(self.model_main.get_weights())
                    self.model_target.save(self.names["save_model"])

            # END TRAINING
            if self.episodes_count > 20000:
                for pipe in self.pipes :
                    pipe.send("EXIT")
                break

