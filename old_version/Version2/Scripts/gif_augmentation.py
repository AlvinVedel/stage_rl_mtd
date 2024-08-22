import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


input_path = "animations_inference/episode2.gif"
output_path = "gif_augmented.gif"
new_size = (100, 100)


def resize_gif(input_path, output_path, new_size):
    # Ouvrir le GIF avec PIL
    gif = Image.open(input_path)

    # Redimensionner chaque frame du GIF
    resized_frames = []
    for frame in range(gif.n_frames):
        gif.seek(frame)
        resized_frame = gif.resize(new_size)
        resized_frames.append(resized_frame)

    # Sauvegarder le GIF redimensionné
    resized_frames[0].save(output_path, save_all=True, append_images=resized_frames[1:], loop=0, duration=100)


resize_gif(input_path, output_path, new_size)



