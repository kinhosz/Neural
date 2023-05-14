from numba import cuda
from ..lib.GPU import *

def layer(x, w, b):
    LEN = w.shape[1]

    arr_dvc = cuda.to_device(np.zeros([1, LEN]))
    x_dvc = cuda.to_device(x)
    w_dvc = cuda.to_device(w)
    b_dvc = cuda.to_device(b)

    dotMatrix[kernelConfig1D(LEN)](arr_dvc, x_dvc, w_dvc, b_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr