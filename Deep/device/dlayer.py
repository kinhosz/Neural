from numba import cuda
from ..lib.GPU import *

def dlayer(w, alpha):
    LEN = w.shape[0]
    LEN2 = w.shape[1]

    w_dvc = cuda.to_device(w)
    alpha_dvc = cuda.to_device(alpha)
    arr_dvc = cuda.to_device(np.zeros([1, LEN]))

    dotMatrix_derivate[kernelConfig3D(1, LEN, LEN2)](arr_dvc, w_dvc, alpha_dvc)
    cuda.synchronize()

    arr = arr_dvc.copy_to_host()
    return arr