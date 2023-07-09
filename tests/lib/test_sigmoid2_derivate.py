import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_sigmoid2_derivate_gpu():
    BATCH = 512
    DIM = 1000
    
    signals_host = np.random.randn(BATCH, 1, DIM)
    alphas_host = np.random.randn(BATCH, 1, DIM)
    
    signals_dvc = cuda.to_device(signals_host)
    alphas_dvc = cuda.to_device(alphas_host)
    buffer = cuda.device_array(shape=(BATCH, 1, DIM), dtype=np.float64)
    
    ret_cpu = cpu.sigmoid2_derivate(signals_host, alphas_host)
    gpu.sigmoid2_derivate(signals=signals_dvc, alphas=alphas_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM):
            assert abs(ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) < EPS
