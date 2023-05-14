from numba import cuda
from ..lib.GPU import *

def loss(predicted, target):
    predicted_dvc = cuda.to_device(predicted)
    target_dvc = cuda.to_device(target)
    arr_dvc = cuda.to_device(np.zeros(1))

    mse[kernelConfig1D(predicted_dvc.shape[0])](arr_dvc, predicted_dvc, target_dvc)
    cuda.synchronize()
    arr = arr_dvc.copy_to_host()
    return arr