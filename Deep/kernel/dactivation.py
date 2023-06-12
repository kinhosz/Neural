from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def dactivation(z, alpha, buffer):
    LEN = z.shape[1]

    t = timer()
    z_dvc, alpha_dvc = loadTo(z, alpha, mode='GPU')
    t = timer() - t
    add(t)

    sigmoid2_derivate[kernelConfig1D(LEN)](buffer, z_dvc, alpha_dvc)
    cuda.synchronize()

    return buffer