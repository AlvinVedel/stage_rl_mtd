import numpy as np
import cupy as cp
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# Créer un tableau numpy de taille 400x400
array_cpu = np.random.rand(400, 400, 6).astype(np.float32)

# Calculer la taille du tableau en bits
size_in_bits = array_cpu.nbytes * 8
print(f"Size of array in bits: {size_in_bits}")

# Mesurer le temps de transfert CPU -> GPU

liste_transfert = []
liste_compute = []

for i in range(128) :
    start_time = time.time()
    array_gpu = cp.asarray(array_cpu)
    end_time = time.time()
    transfer_time = end_time - start_time
    liste_transfert.append(transfer_time)
    start_compute_time = time.time()
    result_gpu = cp.linalg.norm(array_gpu)
    end_compute_time = time.time()
    compute_time = end_compute_time - start_compute_time
    liste_compute.append(compute_time)

print(f"Transfer time (CPU to GPU): {np.mean(liste_transfert)} seconds")

# Pour la comparaison du temps de calcul, effectuer une opération sur le GPU


print(f"Compute time on GPU: {np.mean(liste_compute)} seconds")
