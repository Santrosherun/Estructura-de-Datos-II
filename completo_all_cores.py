import os
import time
import multiprocessing

# Asignación de todos los núcleos disponibles
total_cores = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = total_cores
os.environ['OPENBLAS_NUM_THREADS'] = total_cores
os.environ['MKL_NUM_THREADS'] = total_cores
os.environ['VECLIB_MAXIMUM_THREADS'] = total_cores
os.environ['NUMEXPR_NUM_THREADS'] = total_cores

import numpy as np

def flujo_multi_nucleo(N=10000):
    print(f"Multiplicación optimizada para {total_cores} NÚCLEOS...")
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
    
    # En multihilo, process_time suma el tiempo de cada núcleo por separado, 
    # por lo que despejamos t_waiting usando el tiempo de reloj real para la multiplicación
    tiempo_real_multiplicacion = t_respuesta - t_mem - t_io
    t_waiting = max(0, t_respuesta - (tiempo_real_multiplicacion + t_mem + t_io))

    print(f"T_CPU (Suma de todos los hilos): {t_cpu:.4f}s")
    print(f"T_memory: {t_mem:.4f}s | T_I/O: {t_io:.4f}s | T_waiting: {t_waiting:.4f}s")
    print(f"T_respuesta total: {t_respuesta:.4f}s")

    if os.path.exists('resultado_temp.npy'):
        os.remove('resultado_temp.npy')

if __name__ == "__main__":
    flujo_multi_nucleo()
