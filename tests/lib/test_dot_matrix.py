from Kinho.lib import cpu, gpu
from numba import cuda
import numpy as np

EPS = 1e-8

def test_dot_matrix_gpu():
    BATCH = 64
    N = 100
    M = 200

    signals_host = np.random.randn(BATCH, 1, N)
    weight_host = np.random.randn(N, M)
    bias_host = np.random.randn(1, M)
    
    signals_device = cuda.to_device(signals_host)
    weight_device = cuda.to_device(weight_host)
    bias_device = cuda.to_device(bias_host)
    buffer_device = cuda.device_array(shape=(BATCH, 1, M), dtype=np.float64)
    
    ret_cpu = cpu.dot_matrix(signals=signals_host, weight=weight_host, bias=bias_host)
    gpu.dot_matrix(signals=signals_device, weight=weight_device, bias=bias_device, buffer=buffer_device)
    ret_gpu = buffer_device.copy_to_host()
    
    for i in range(BATCH):
        for j in range(M):
            assert abs(ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) < EPS
