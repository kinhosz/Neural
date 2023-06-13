from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def selector(z, res, buffer):
    LEN = z.shape[1]

    t = timer()
    z_dvc, res_dvc = loadTo(z, res, mode='GPU')
    memset[kernelConfig1D(1)](res_dvc)
    cuda.synchronize()
    t = timer() - t
    add(t)

    softmax_p1[kernelConfig1D(LEN)](buffer, z_dvc, res_dvc)
    cuda.synchronize()

    softmax_p2[kernelConfig1D(LEN)](buffer, res_dvc)
    cuda.synchronize()

    return buffer