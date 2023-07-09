import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_mse_derivate_gpu():
    BATCH = 256
    DIM = 40
    
    predict_host = np.random.randn(BATCH, 1, DIM)
    target_host = np.random.randn(BATCH, 1, DIM)
    buffer_host = np.random.randn(BATCH, 1, DIM)
    
    predict_dvc = cuda.to_device(predict_host)
    target_dvc = cuda.to_device(target_host)
    buffer_dvc = cuda.to_device(buffer_host)
    
    ret_cpu = cpu.mse_derivate(predict_host, target_host)
    gpu.mse_derivate(predict_dvc, target_dvc, buffer_dvc)
    ret_gpu = buffer_dvc.copy_to_host()
    
    for i in range(BATCH):
        for j in range(DIM):
            assert (ret_cpu[i, 0, j] - ret_gpu[i, 0, j]) <= EPS
