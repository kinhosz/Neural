import numpy as np
from numba import cuda
from Kinho.lib import cpu, gpu

EPS = 1e-8

def test_bce_gpu():
    DIM = 100
    BATCH = 4
    
    predict_host = np.random.rand(BATCH, 1, DIM)
    target_host = np.random.rand(BATCH, 1, DIM)
    
    predict_dvc = cuda.to_device(predict_host)
    target_dvc = cuda.to_device(target_host)
    buffer = cuda.device_array(shape=(1,), dtype=np.float64)
    
    ret_cpu = cpu.bce(predict=predict_host, target=target_host)
    gpu.bce(predict=predict_dvc, target=target_dvc, buffer=buffer)
    ret_gpu = buffer.copy_to_host()
    
    assert abs(ret_cpu[0] - ret_gpu[0]) < EPS
