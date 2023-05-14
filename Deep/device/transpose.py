from numba import cuda
from ..lib.GPU import *

def transpose(z, derror):
    LEN1 = z.shape[1]
    LEN2 = derror.shape[1]

    arr_dvc = cuda.to_device(np.zeros([LEN1, LEN2]))
    x_dvc = cuda.to_device(z)
    derror_dvc = cuda.to_device(derror)

    transposeDot[kernelConfig2D(LEN1, LEN2)](arr_dvc, x_dvc, derror_dvc)
    cuda.synchronize()

    nabla_w = arr_dvc.copy_to_host()
    return nabla_w