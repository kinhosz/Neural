import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_stochastic_gradient_descent_gpu():
    BATCH = 256
    DIM = 100
    DIM2 = 200
    
    gradients_host = np.random.randn(BATCH, DIM, DIM2)
    
    gradients_dvc = cuda.to_device(gradients_host)
    buffer = cuda.device_array(shape=(DIM, DIM2), dtype=np.float64)
    
    ret_cpu = cpu.stochastic_gradient_descent(gradients=gradients_host)
    gpu.stochastic_gradient_descent(gradients=gradients_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    for i in range(DIM):
        for j in range(DIM2):
            assert abs(ret_cpu[i, j] - ret_gpu[i, j]) < EPS
