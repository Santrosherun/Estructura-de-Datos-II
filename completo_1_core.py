import os
import time
import gc

# 1. Restricción estricta a 1 núcleo
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 2. Anclar el proceso estrictamente al núcleo 0
try:
    os.sched_setaffinity(0, {0})
except AttributeError:
    pass

import numpy as np

def flujo_un_nucleo_ciclo(N=10000, M=10):
    tiempos_cpu, tiempos_mem, tiempos_io, tiempos_espera, tiempos_respuesta = [], [], [], [], []
    print(f"Evaluación 1 NÚCLEO (Anclado al Core 0) | N={N}, M={M}...")
    rng = np.random.default_rng()

    for i in range(M):
        t_inicio_total = time.time()

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

        t_respuesta = time.time() - t_inicio_total
        t_espera = max(0, t_respuesta - (t_cpu + t_mem + t_io))

        tiempos_cpu.append(t_cpu)
        tiempos_mem.append(t_mem)
        tiempos_io.append(t_io)
        tiempos_espera.append(t_espera)
        tiempos_respuesta.append(t_respuesta)

        os.remove('resultado_temp.npy')
        del matriz_a, matriz_b, resultado
        gc.collect()

    print(f"T_CPU prom: {np.mean(tiempos_cpu):.4f}s | T_mem prom: {np.mean(tiempos_mem):.4f}s")
    print(f"T_I/O prom: {np.mean(tiempos_io):.4f}s | T_wait prom: {np.mean(tiempos_espera):.4f}s")
    print(f"T_respuesta total prom: {np.mean(tiempos_respuesta):.4f}s")

if __name__ == "__main__":
    flujo_un_nucleo_ciclo()
