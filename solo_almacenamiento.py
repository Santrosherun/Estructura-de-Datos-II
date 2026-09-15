import time
import numpy as np

def medir_almacenamiento(N=10000):
    print(f"Midiendo SOLO ALMACENAMIENTO (I/O) para matriz de {N}x{N}...")
    matriz = np.random.default_rng().random((N, N), dtype=np.float32)
    
    t_inicio = time.perf_counter()
    np.save('prueba_io.npy', matriz)
    t_io = time.perf_counter() - t_inicio
    
    print(f"T_I/O (Tiempo de escritura en disco): {t_io:.4f}s")

if __name__ == "__main__":
    medir_almacenamiento()