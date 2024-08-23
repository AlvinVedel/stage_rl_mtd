from processes.thread_process import DistributionalThreadProcess
from models.variational_models.dueling_qr_dqn_varia import Dueling_QRDQN_varia1_model2, Dueling_QRDQN_varia2_model2, Dueling_QRDQN_varia3_model2
from models.dueling_qr_dqn import Dueling_QRDQN_model2
import threading
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


models = [Dueling_QRDQN_varia1_model2(variational_dim=1024, n_quantiles=51), 
          Dueling_QRDQN_varia2_model2(variational_dim=1024, n_quantiles=51), 
          Dueling_QRDQN_varia3_model2(variational_dim=1024, n_quantiles=51)]


list_names = [{
    "gateway" : "gateway_DoubleTrain_DistrVaeV1_1024.png",
    "step" : "steps_DoubleTrain_DistrVaeV1_1024.png",
    "lap" : "laps_DoubleTrain_DistrVaeV1_1024 .png",
    "save_model":"speeded_up_doubleTrain_DistrVaeV1_1024.h5",
    "reward" : "reward_DoubleTrain_DistrVaeV1_1024 .png"
},
{
    "gateway" : "gateway_DoubleTrain_DistrVaeV2_1024 .png",
    "step" : "steps_DoubleTrain_DistrVaeV2_1024 .png",
    "lap" : "laps_DoubleTrain_DistrVaeV2_1024 .png",
    "save_model":"speeded_up_doubleTrain_DistrVaeV2_1024 .h5",
    "reward" : "reward_DoubleTrain_DistrVaeV2_1024 .png"
},
{
    "gateway" : "gateway_DoubleTrain_DistrVaeV3_1024 .png",
    "step" : "steps_DoubleTrain_DistrVaeV3_1024 .png",
    "lap" : "laps_DoubleTrain_DistrVaeV3_1024 .png",
    "save_model":"speeded_up_doubleTrain_DistrVaeV3_1024 .h5",
    "reward" : "reward_DoubleTrain_DistrVaeV3_1024 .png"
}]


thread_list = []
gpu_lock = threading.Lock()


t1 = DistributionalThreadProcess(0, models[0], gpu_lock, list_names[0], 4, 4, 4, 51)
t1.start()
thread_list.append(t1)
t2 = DistributionalThreadProcess(1, models[1], gpu_lock, list_names[1], 4, 4, 4, 51)
t2.start()
thread_list.append(t2)
t3 = DistributionalThreadProcess(2, models[2], gpu_lock, list_names[2], 4, 4, 4, 51)
t3.start()
thread_list.append(t3)

for t in thread_list :
    t.join()

print("TRAINING ENDED")



