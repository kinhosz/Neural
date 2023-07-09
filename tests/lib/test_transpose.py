import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_transpose_gpu():
    BATCH = 512
    DIM1 = 100
    DIM2 = 300
    
    signals_host = np.random.randn(BATCH, 1, DIM1)
    alphas_host = np.random.randn(BATCH, 1, DIM2)
    
    signals_dvc = cuda.to_device(signals_host)
    alphas_dvc = cuda.to_device(alphas_host)
    buffer = cuda.device_array(shape=(BATCH, DIM1, DIM2), dtype=np.float64)
    
    ret_cpu = cpu.transpose(signals=signals_host, alphas=alphas_host)
    gpu.transpose(signals=signals_dvc, alphas=alphas_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM1):
            for k in range(DIM2):
                assert abs(ret_cpu[i, j, k] - ret_gpu[i, j, k]) < EPS
