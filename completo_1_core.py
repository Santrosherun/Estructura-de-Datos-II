import os
import time

# Restricción estricta a 1 núcleo antes de importar numpy
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np

def flujo_un_nucleo(N=10000):
    print("Multiplicación forzada a 1 NÚCLEO...")
    t_inicio = time.time()
    rng = np.random.default_rng()

    t_in_mem = time.perf_counter()
    matriz_a = rng.random((N, N), dtype=np.float32) * 2 - 1
    matriz_b = rng.random((N, N), dtype=np.float32) * 2 - 1
    t_mem = time.perf_counter() - t_in_mem

    t_in_cpu = time.process_time()
    resultado = np.dot(matriz_a, matriz_b)
    t_cpu = time.process_time() - t_in_cpu

    t_in_io = time.perf_counter()
    np.save('resultado_temp.npy', resultado)
    t_io = time.perf_counter() - t_in_io

    t_respuesta = time.time() - t_inicio
    t_waiting = max(0, t_respuesta - (t_cpu + t_mem + t_io))

    print(f"T_CPU: {t_cpu:.4f}s | T_memory: {t_mem:.4f}s")
    print(f"T_I/O: {t_io:.4f}s | T_waiting: {t_waiting:.4f}s")
    print(f"T_respuesta total: {t_respuesta:.4f}s")
    
    if os.path.exists('resultado_temp.npy'):
        os.remove('resultado_temp.npy')

if __name__ == "__main__":
    flujo_un_nucleo()
