from numba import cuda
from ..lib.GPU import *

def dloss(predicted, target):
    LEN = predicted.shape[1]
    predicted_dvc = cuda.to_device(predicted)
    target_dvc = cuda.to_device(target)
    arr_dvc = cuda.to_device(np.zeros([1, LEN]))

    mse_derivate[kernelConfig1D(LEN)](arr_dvc, predicted_dvc, target_dvc)
    cuda.synchronize()
    arr = arr_dvc.copy_to_host()
    return arr