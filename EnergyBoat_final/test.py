import numpy as np
import tensorflow as tf
import os
import threading
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from processes.thread_process import InferenceThreadProcess

model = tf.keras.models.load_model("../trained_models/speeded_up_doubleTrain_dueling_qr.h5")
names = {"gif_name" : "gif_inference", "metrics_name":"data_dueling_qr.csv"}

gpu_lock = threading.Lock()
t = InferenceThreadProcess(0, model=model, gpu_lock=gpu_lock, names=names, distributional=True, nb_inferences=1, save_metrics=True, render=True)
t.start()
t.join()



