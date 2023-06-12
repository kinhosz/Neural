from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def loss(predicted, target, buffer):
    t = timer()
    predicted_dvc, target_dvc = loadTo(predicted, target, mode='GPU')
    t = timer() - t
    add(t)

    mse[kernelConfig1D(predicted_dvc.shape[0])](buffer, predicted_dvc, target_dvc)
    cuda.synchronize()

    return buffer