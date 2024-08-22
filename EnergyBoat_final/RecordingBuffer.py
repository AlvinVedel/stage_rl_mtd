import numpy as np
import matplotlib.pyplot as plt

class RecordingBuffer():
    def __init__(self, names, averaging_frequency=20, plot_frequency=200, directory='./'):
        self.averaging_frequency = averaging_frequency
        self.plot_frequency = plot_frequency
        self.directory = directory
        self.names = names
        self.gateway_record_transition = []
        self.gateway_record = []
        self.laps_record_transition = []
        self.laps_record = []
        self.steps_record_transition = []
        self.steps_record = []
        self.reward_record_transition = []
        self.reward_record = []
        self.record_count = 0

    def record(self, nb_gateways, nb_laps, nb_steps, sum_reward) :
        self.gateway_record_transition.append(nb_gateways)
        self.laps_record_transition.append(nb_laps)
        self.steps_record_transition.append(nb_steps)
        self.reward_record_transition.append(sum_reward)
        self.record_count+=1
        if self.record_count % self.averaging_frequency == 0 :
            self.averaging()

    def averaging(self):
        self.gateway_record.append(np.mean(self.gateway_record_transition))
        self.laps_record.append(np.mean(self.laps_record_transition))
        self.steps_record.append(np.mean(self.steps_record_transition))
        self.reward_record.append(np.mean(self.reward_record_transition))
        self.gateway_record_transition = []
        self.laps_record_transition = []
        self.steps_record_transition = []
        self.reward_record_transition = []
        if self.record_count % self.plot_frequency == 0 :
            self.plot()

    def plot(self) :
        xlabel = 'Episodes (x'+str(self.averaging_frequency)+')'

        x = list(range(len(self.gateway_record)))
        plt.plot(x, self.gateway_record, marker='o', linestyle='-', color='b')
        plt.xlabel(xlabel)
        plt.ylabel('Nombre de points de passage')
        plt.title('Nombre de points de passage atteint en moyenne pour '+str(self.averaging_frequency)+' épisodes')
        plt.grid(True)
        plt.savefig(self.names["gateway"])
        plt.close()

        plt.plot(x, self.reward_record, marker='o', linestyle='-', color='b')
        plt.xlabel(xlabel)
        plt.ylabel('Somme des rewards')
        plt.title('Somme des rewards par moyenné sur '+str(self.averaging_frequency)+' épisodes')
        plt.grid(True)
        plt.savefig(self.names["reward"])
        plt.close()

        new_metric = np.array(self.steps_record)/ (1+np.array(self.gateway_record))
        plt.plot(x, self.steps_record, linestyle='-', color='r', alpha=0.5, label='steps vivants')
        plt.plot(x, new_metric, linestyle='-', color='b', label='steps par objectif')
        plt.xlabel(xlabel)
        plt.ylabel('Nombre de pas')
        plt.title("Nombre de steps moyen et rapporté au nombre d'objectifs atteints")
        plt.grid(True)
        plt.legend()
        plt.savefig(self.names["steps"])
        plt.close()

        new_metric = np.array(self.steps_record)/ 10
        plt.plot(x, np.array(self.laps_record)*10, marker='o', linestyle='-', color='b')
        plt.plot(x, new_metric, linestyle='-', color='r', label='steps')
        plt.xlabel(xlabel)
        plt.ylabel('Nombre de tours')
        plt.title('Nombre de tours(x10) et nombre de pas (/10)')
        plt.grid(True)
        plt.legend()
        plt.savefig(self.names["laps"])
        plt.close()