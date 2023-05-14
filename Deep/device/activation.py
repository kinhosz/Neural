from numba import cuda
from ..lib.GPU import *

def activation(z):
    LEN = z.shape[1]

    z_dvc = cuda.to_device(z)
    arr_dvc = cuda.to_device(np.random.randn(1, LEN))

    sigmoid2[kernelConfig1D(LEN)](arr_dvc, z_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr