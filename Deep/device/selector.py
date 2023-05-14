from numba import cuda
from ..lib.GPU import *

def selector(z):
    LEN = z.shape[1]
    z_dvc = cuda.to_device(z)
    arr_dvc = cuda.to_device(np.random.randn(1, LEN))
    res_dvc = cuda.to_device(np.zeros(1))

    softmax_p1[kernelConfig1D(LEN)](arr_dvc, z_dvc, res_dvc)
    cuda.synchronize()
    softmax_p2[kernelConfig1D(LEN)](arr_dvc, res_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr