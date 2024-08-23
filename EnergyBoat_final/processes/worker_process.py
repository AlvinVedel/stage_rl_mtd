import multiprocessing as mp
import numpy as np
import random
from EnergyBoatScenario.env import EnergyBoatEnv


class WorkerProcess(mp.Process):
    def __init__(self, worker_id, conn, seed, nb_envs, nb_agents, env_config):
        super().__init__()
        self.worker_id = worker_id
        self.conn = conn
        self.seed = seed
        self.nb_envs = nb_envs
        self.nb_agents = nb_agents
        self.env_config = env_config
        self.local_seeds = [random.randint(0, 2**32 - 1) for _ in range(self.nb_envs)]
        self.local_envs = []
        for i in range(self.nb_envs) :
            env = EnergyBoatEnv(env_config, nb_agents=self.nb_agents, init_seed=self.local_seeds[i])
            self.local_envs.append(env)

    def run(self):
        while True:
            task = self.conn.recv()  

            if isinstance(task, str):
                if task == 'EXIT':
                    break
                elif task=='RESET' :
                    states1 = np.zeros((self.nb_envs, self.nb_agents, 220))
                    states2 = np.zeros((self.nb_envs, self.nb_agents, 6))
                    for i, env in enumerate(self.local_envs) :
                        state = env.reset()  
                        states1[i] = state[0]
                        states2[i] = state[1]
                    initial_states = [self.worker_id, states1, states2]
                    self.conn.send(initial_states)



            elif isinstance(task, tuple) and len(task) == 2 and task[0] == "RESET":
                nb_gateways, nb_laps, nb_steps, sum_reward = self.local_envs[task[1]].get_metriques()
                env_infos = [nb_gateways, nb_laps, nb_steps, sum_reward]

                state = self.local_envs[task[1]].reset()

                results = [state, env_infos]
                self.conn.send(results)



            else :
                state1_buffer = np.zeros((self.nb_envs, self.nb_agents, 220))
                state2_buffer = np.zeros((self.nb_envs, self.nb_agents, 6))
                rl_buffer = np.zeros((self.nb_envs, self.nb_agents, 3))
                steps_counter = np.zeros(self.nb_envs)
                for i in range(self.nb_envs) :
                    next_state, reward, dejadone, done, useless1, useless2 = self.local_envs[i].step(task[i])
                    state1_buffer[i] = next_state[0] 
                    state2_buffer[i] = next_state[1]  
                    rl_buffer[i] = np.concatenate([np.expand_dims(reward, axis=1), np.expand_dims(done, axis=1), np.expand_dims(dejadone, axis=1)], axis=-1) 
                    steps_counter[i] = np.max(env.nb_steps)  
                    
                obs = [self.worker_id, state1_buffer, state2_buffer, rl_buffer, steps_counter] 
                self.conn.send(obs)

    
