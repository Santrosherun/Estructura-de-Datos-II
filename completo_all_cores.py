import os
import time
import gc
import multiprocessing
import numpy as np

# Asignación a todos los núcleos físicos detectados
total_cores = str(multiprocessing.cpu_count())
os.environ['OMP_NUM_THREADS'] = total_cores
os.environ['OPENBLAS_NUM_THREADS'] = total_cores
os.environ['MKL_NUM_THREADS'] = total_cores
os.environ['VECLIB_MAXIMUM_THREADS'] = total_cores
os.environ['NUMEXPR_NUM_THREADS'] = total_cores

def flujo_multi_nucleo_ciclo(N=10000, M=10):
    tiempos_cpu_suma, tiempos_mem, tiempos_io, tiempos_espera, tiempos_respuesta = [], [], [], [], []
    print(f"Evaluación TODOS LOS NÚCLEOS ({total_cores}) | N={N}, M={M}...")
    rng = np.random.default_rng()

    for i in range(M):
        t_inicio_total = time.time()

        t_in_mem = time.perf_counter()
        matriz_a = rng.random((N, N), dtype=np.float32) * 2 - 1
        matriz_b = rng.random((N, N), dtype=np.float32) * 2 - 1
        t_mem = time.perf_counter() - t_in_mem

        # Medimos tanto la suma de hilos (process_time) como el reloj real (time)
        t_inicio_cpu_reloj = time.time()
        t_inicio_cpu_suma = time.process_time()
        resultado = np.dot(matriz_a, matriz_b)
        t_cpu_suma = time.process_time() - t_inicio_cpu_suma
        t_cpu_reloj = time.time() - t_inicio_cpu_reloj

        t_in_io = time.perf_counter()
        np.save('resultado_temp.npy', resultado)
        t_io = time.perf_counter() - t_in_io

        t_respuesta = time.time() - t_inicio_total
        # Despejamos usando el tiempo de reloj real de la CPU para mantener coherencia física
        t_espera = max(0, t_respuesta - (t_cpu_reloj + t_mem + t_io))

        tiempos_cpu_suma.append(t_cpu_suma)
        tiempos_mem.append(t_mem)
        tiempos_io.append(t_io)
        tiempos_espera.append(t_espera)
        tiempos_respuesta.append(t_respuesta)

        os.remove('resultado_temp.npy')
        del matriz_a, matriz_b, resultado
        gc.collect()

    print(f"T_CPU (Suma Hilos) prom: {np.mean(tiempos_cpu_suma):.4f}s")
    print(f"T_mem prom: {np.mean(tiempos_mem):.4f}s | T_I/O prom: {np.mean(tiempos_io):.4f}s")
    print(f"T_wait prom: {np.mean(tiempos_espera):.4f}s | T_respuesta prom: {np.mean(tiempos_respuesta):.4f}s")

if __name__ == "__main__":
    flujo_multi_nucleo_ciclo()
