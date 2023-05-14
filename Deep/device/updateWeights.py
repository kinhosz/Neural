from numba import cuda
from ..lib.GPU import *

def updateWeight(weights, eta, nabla_w):
    LEN1, LEN2 = weights.shape

    eta_dvc = cuda.to_device(eta)
    nabla_w_dvc = cuda.to_device(nabla_w)
    w_dvc = cuda.to_device(weights)

    updateWeights[kernelConfig2D(LEN1, LEN2)](w_dvc, eta_dvc, nabla_w_dvc)
    cuda.synchronize()

    weights = w_dvc.copy_to_host()
    return weights