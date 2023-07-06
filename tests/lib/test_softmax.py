from Kinho.lib import cpu, gpu
import numpy as np
from numba import cuda

EPS = 1e-8

def test_softmax_gpu():
    BATCH = 128
    DIM = 40
    
    signals_host = np.random.randn(BATCH, 1, DIM)
    extra = np.empty(shape=(BATCH, 1))
    buffer = np.empty(shape=(BATCH, 1, DIM))
    
    signals_dvc = cuda.to_device(signals_host)
    extra_dvc = cuda.to_device(extra)
    buffer_dvc = cuda.to_device(buffer)
    
    ret_cpu = cpu.softmax(signals_host)
    gpu.softmax(signals=signals_dvc, extra=extra_dvc, buffer=buffer_dvc)
    ret_gpu = buffer_dvc.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM):
            assert abs(ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) < EPS
