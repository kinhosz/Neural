import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_dot_matrix_derivate_gpu():
    BATCH = 512
    DIM1 = 100
    DIM2 = 300
    
    weight_host = np.random.randn(DIM1, DIM2)
    alphas_host = np.random.randn(BATCH, 1, DIM2)
    
    weight_dvc = cuda.to_device(weight_host)
    alphas_dvc = cuda.to_device(alphas_host)
    buffer = cuda.device_array(shape=(BATCH, 1, DIM1), dtype=np.float64)
    
    ret_cpu = cpu.dot_matrix_derivate(const_matrix=weight_host, alphas=alphas_host)
    gpu.dot_matrix_derivate(const_matrix=weight_dvc, alphas=alphas_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM1):
            assert abs(ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) < EPS
