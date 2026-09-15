import time
import numpy as np
import os

# Forzar 1 core para la prueba aislada
os.environ['OMP_NUM_THREADS'] = '1'

def medir_multiplicacion(N=10000):
    print(f"Midiendo SOLO MULTIPLICACIÓN (CPU) para matrices de {N}x{N}...")
    rng = np.random.default_rng()
    matriz_a = rng.random((N, N), dtype=np.float32)
    matriz_b = rng.random((N, N), dtype=np.float32)
    
    t_inicio = time.process_time()
    resultado = np.dot(matriz_a, matriz_b)
    t_cpu = time.process_time() - t_inicio
    
    print(f"T_CPU (Tiempo de procesamiento): {t_cpu:.4f}s")

if __name__ == "__main__":
    medir_multiplicacion()