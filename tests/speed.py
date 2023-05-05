import numpy as np
from numba import cuda
from Deep.lib.GPU import *
from .utils import *
from timeit import default_timer as timer

ALPHA = 1.5
MS_LIMIT = 20.0

def dotMatrix_test(LEN_ARRAY):
    LEN_ARRAY1 = int(math.sqrt(LEN_ARRAY))
    LEN_ARRAY2 = LEN_ARRAY1
    
    A_host = np.random.randn(1, LEN_ARRAY1)
    B_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
    C_host = np.random.randn(1, LEN_ARRAY2)

    A_device = cuda.to_device(A_host)
    B_device = cuda.to_device(B_host)
    C_device = cuda.to_device(C_host)

    arr_host = np.random.randn(1, LEN_ARRAY2)
    arr_device = cuda.to_device(arr_host)

    LOOP = 100

    cpu_timer = timer()
    for i in range(LOOP):
        dotMatrix_cpu(A_host, B_host, C_host)
    cpu_timer = timer() - cpu_timer

    gpu_timer = timer()
    for i in range(LOOP):
        dotMatrix[kernelConfig2D(1, LEN_ARRAY1)](arr_device, A_device, B_device, C_device)
        cuda.synchronize()

    gpu_timer = timer() - gpu_timer

    cpu_timer *= 1000
    gpu_timer *= 1000

    cpu_timer /= LOOP
    gpu_timer /= LOOP

    return gpu_timer < MS_LIMIT

def dotMatrix_speed_1e2():
    return dotMatrix_test(1e2)

def dotMatrix_speed_1e3():
    return dotMatrix_test(1e3)

def dotMatrix_speed_1e4():
    return dotMatrix_test(1e4)

def dotMatrix_speed_1e5():
    return dotMatrix_test(1e5)

def dotMatrix_speed_1e6():
    return dotMatrix_test(1e6)

def dotMatrix_speed_1e7():
    return dotMatrix_test(1e7)

def dotMatrix_speed_1e8():
    return dotMatrix_test(1e8)

def compiler(tests):
    for t in tests:
        t()

def test():
    tests = [dotMatrix_speed_1e2, dotMatrix_speed_1e3, dotMatrix_speed_1e4, dotMatrix_speed_1e5,
             dotMatrix_speed_1e6, dotMatrix_speed_1e7, dotMatrix_speed_1e8]
    
    compiler(tests)
    
    logger("speed", tests)

if __name__ == "__main__":
    test()