from numba import cuda
from ..lib.GPU import *

def dselector(z, alpha):
    LEN = z.shape[1]

    z_dvc = cuda.to_device(z)
    alpha_dvc = cuda.to_device(alpha)
    arr_dvc = cuda.to_device(np.random.randn(1, LEN))
    ss_dvc = cuda.to_device(np.zeros(1))
    st_dvc = cuda.to_device(np.zeros(1))

    softmax_sum_derivate[kernelConfig1D(LEN)](arr_dvc, z_dvc, alpha_dvc, ss_dvc, st_dvc)
    cuda.synchronize()
    arr = arr_dvc.copy_to_host()
    softmax_derivate[kernelConfig1D(LEN)](arr_dvc, alpha_dvc, ss_dvc, st_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr
