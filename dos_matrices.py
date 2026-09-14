import time
import numpy as np

def medir_dos_matrices(N=10000):
    print(f"Creando DOS matrices de {N}x{N}...")
    t_inicio = time.time()
    
    t_in_mem = time.perf_counter()
    rng = np.random.default_rng()
    matriz_a = rng.random((N, N), dtype=np.float32) * 2 - 1
    matriz_b = rng.random((N, N), dtype=np.float32) * 2 - 1
    t_mem = time.perf_counter() - t_in_mem
    
    t_respuesta = time.time() - t_inicio
    t_waiting = max(0, t_respuesta - t_mem)
    
    print(f"T_memory: {t_mem:.4f}s | T_waiting: {t_waiting:.4f}s | T_respuesta: {t_respuesta:.4f}s")

if __name__ == "__main__":
    medir_dos_matrices()
