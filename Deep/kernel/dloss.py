from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def dloss(predicted, target, buffer):
    LEN = predicted.shape[1]

    t = timer()
    predicted_dvc, target_dvc = loadTo(predicted, target, mode='GPU')
    t = timer() - t
    add(t)

    mse_derivate[kernelConfig1D(LEN)](buffer, predicted_dvc, target_dvc)
    cuda.synchronize()

    return buffer