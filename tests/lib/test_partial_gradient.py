import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_partial_gradient_gpu():
    DIM1 = 100
    DIM2 = 200
    
    weight_host = np.random.randn(DIM1, DIM2)
    gradient_host = np.random.randn(DIM1, DIM2)
    eta_host = np.random.randn(1)
    
    weight_dvc = cuda.to_device(weight_host)
    gradient_dvc = cuda.to_device(gradient_host)
    eta_dvc = cuda.to_device(eta_host)
    
    ret_cpu = cpu.partial_gradient(weight=weight_host, eta=eta_host, gradient=gradient_host)
    gpu.partial_gradient(weight=weight_dvc, eta=eta_dvc, gradient=gradient_dvc)
    ret_gpu = weight_dvc.copy_to_host()
    
    for i in range(DIM1):
        for j in range(DIM2):
            assert abs(ret_cpu[i, j] - ret_gpu[i, j]) < EPS
