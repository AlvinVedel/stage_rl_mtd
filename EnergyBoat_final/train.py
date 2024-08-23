from processes.thread_process import DistributionalThreadProcess
from models.dueling_qr_dqn import Dueling_QRDQN_model1, Dueling_QRDQN_model2
from models.dueling_qr_dqn import Dueling_QRDQN_model2
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


models = [Dueling_QRDQN_model1(n_quantiles=51), 
          Dueling_QRDQN_model2(n_quantiles=51)]


list_names = [{
    "gateway" : "gateway_DoubleTrain_model1.png",
    "step" : "steps_DoubleTrain_model1.png",
    "lap" : "laps_DoubleTrain_model1.png",
    "save_model":"speeded_up_doubleTrain_model1.h5",
    "reward" : "reward_DoubleTrain_model1.png"
},
{
    "gateway" : "gateway_DoubleTrain_model2.png",
    "step" : "steps_DoubleTrain_model2.png",
    "lap" : "laps_DoubleTrain_model2.png",
    "save_model":"speeded_up_doubleTrain_model2.h5",
    "reward" : "reward_DoubleTrain_model2.png"
}]


thread_list = []
gpu_lock = threading.Lock()


t1 = DistributionalThreadProcess(0, models[0], gpu_lock, list_names[0], 4, 4, 4, 51)
t1.start()
thread_list.append(t1)
t2 = DistributionalThreadProcess(1, models[1], gpu_lock, list_names[1], 4, 4, 4, 51)
t2.start()
thread_list.append(t2)


for t in thread_list :
    t.join()

print("TRAINING ENDED")



