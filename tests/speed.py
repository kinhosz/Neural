import numpy as np
from numba import cuda
from Deep.lib.GPU import *
from .utils import *
from timeit import default_timer as timer

ALPHA = 1.5

def dotMatrix_test(LEN_ARRAY1, LEN_ARRAY2):    
    A_host = np.random.randn(1, LEN_ARRAY1)
    B_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
    C_host = np.random.randn(1, LEN_ARRAY2)

    A_device = cuda.to_device(A_host)
    B_device = cuda.to_device(B_host)
    C_device = cuda.to_device(C_host)

    arr_host = np.random.randn(1, LEN_ARRAY1)
    arr_device = cuda.to_device(arr_host)

    LOOP = 1

    cpu_timer = timer()
    for i in range(LOOP):
        A_host = dotMatrix_cpu(A_host, B_host, C_host)
    cpu_timer = timer() - cpu_timer

    gpu_timer = timer()
    dotMatrix[kernelConfig3D(1, LEN_ARRAY1, LEN_ARRAY1)](arr_device, A_device, B_device)
    cuda.synchronize()
    sum[kernelConfig2D(1, LEN_ARRAY1)](arr_device, C_device)
    cuda.synchronize()
    gpu_timer = timer() - gpu_timer
    
    rate = gpu_timer / cpu_timer

    return rate < ALPHA

def dotMatrix_speed_100():
    dotMatrix_test(100, 200)
    return dotMatrix_test(100, 200)

def dotMatrix_speed_1000():
    dotMatrix_test(1000, 2000)
    return dotMatrix_test(1000, 2000)

def dotMatrix_speed_10000():
    dotMatrix_test(10000, 20000)
    return dotMatrix_test(10000, 20000)

def test():
    tests = [dotMatrix_speed_100, dotMatrix_speed_1000, dotMatrix_speed_10000]
    
    gerador = logger("speed", tests)
    while gerador != None:
        gerador = next(gerador)

if __name__ == "__main__":
    test()