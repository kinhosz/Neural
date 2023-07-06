from Kinho.lib import cpu, gpu
import numpy as np
from numba import cuda

EPS = 1e-8

def test_softmax_derivate_gpu():
    BATCH = 256
    DIM = 100
    
    signals_host = np.random.randn(BATCH, 1, DIM)
    alphas_host = np.random.randn(BATCH, 1, DIM)
    extra_host = np.random.randn(BATCH, 1)
    
    signals_dvc = cuda.to_device(signals_host)
    alphas_dvc = cuda.to_device(alphas_host)
    extra_dvc = cuda.to_device(extra_host)
    buffer = cuda.device_array(shape=(BATCH, 1, DIM), dtype=np.float64)
    
    ret_cpu = cpu.softmax_derivate(signals=signals_host, alphas=alphas_host)
    gpu.softmax_derivate(signals=signals_dvc, alphas=alphas_dvc, extra=extra_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM):
            assert abs(ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) < EPS
