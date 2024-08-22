import numpy as np

class ReplayBuffer():
    def __init__(self, size_max=500000, obs1_dim=220, obs2_dim=6, prioritized=True):
        self.size_max = size_max
        self.memory_count = 0
        self.memory_index = 0
        self.memory_cone = np.zeros((self.size_max, obs1_dim, 2))
        self.memory_sup = np.zeros((self.size_max, obs2_dim*2))
        self.memory_rl = np.zeros((self.size_max, 3))
        self.poids = np.zeros((self.size_max)) 
        self.prioritized = prioritized
        """
        stockage de 'poids' pour implémentation custom du prioritized experience Replay (Schaul et al)
        """

    def store(self, states1, states2, action, reward, done, next_states1, next_states2) :
        """
        Conçu pour stocker un batch de transitions 
        """
        d1 = np.concatenate([np.expand_dims(states1, axis=-1), np.expand_dims(next_states1, axis=-1)], axis=-1)
        d2 = np.concatenate([states2, next_states2], axis=-1)
        d3 = np.concatenate([np.expand_dims(action, axis=1), np.expand_dims(reward, axis=1), np.expand_dims(done, axis=1)], axis=1)
        for i in range(states1.shape[0]) :
            self.memory_cone[self.memory_index] = d1[i]
            self.memory_sup[self.memory_index] = d2[i]
            self.memory_rl[self.memory_index] = d3[i]
            self.poids[self.memory_index] = abs(reward[i])
            self.memory_count = min(self.memory_count+1, self.size_max)
            self.memory_index = (self.memory_index+1)%self.size_max

    def sample(self, batch_size) :
        if self.prioritized :
            indices = np.random.choice(range(self.memory_count), size=batch_size, replace=False)
        else :
            indices = np.random.choice(range(self.memory_count), size=batch_size, replace=False, p=self.poids[:self.memory_count]/np.sum(self.poids[:self.memory_count]))
        co = self.memory_cone[indices]
        su = self.memory_sup[indices]
        rl = self.memory_rl[indices]
        state_samplev = co[..., 0]
        state_next_samplev = co[..., 1]
        state2_samplev = su[..., :6]
        state2_next_samplev = su[..., 6:]
        aa=rl[:, 0]
        rr=rl[:, 1]
        dones=rl[:, 2]
        return state_samplev, state2_samplev, aa, rr, dones, state_next_samplev, state2_next_samplev