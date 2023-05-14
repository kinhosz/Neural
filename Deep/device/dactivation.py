from numba import cuda
from ..lib.GPU import *

def dactivation(z, alpha):
    LEN = z.shape[1]

    z_dvc = cuda.to_device(z)
    alpha_dvc = cuda.to_device(alpha)
    arr_dvc = cuda.to_device(np.zeros([1, LEN]))

    sigmoid2_derivate[kernelConfig1D(LEN)](arr_dvc, z_dvc, alpha_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr