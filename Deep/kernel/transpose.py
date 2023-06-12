from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def transpose(z, derror, buffer):
    LEN1 = z.shape[1]
    LEN2 = derror.shape[1]

    t = timer()
    x_dvc, derror_dvc = loadTo(z, derror, mode='GPU')
    t = timer() - t
    add(t)

    transposeDot[kernelConfig2D(LEN1, LEN2)](buffer, x_dvc, derror_dvc)
    cuda.synchronize()

    return buffer