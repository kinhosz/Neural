from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def dselector(z, alpha, buffer):
    LEN = z.shape[1]

    t = timer()
    z_dvc, alpha_dvc, ss_dvc, st_dvc = loadTo(z, alpha, np.zeros(1), np.zeros(1), mode='GPU')
    t = timer() - t
    add(t)

    softmax_sum_derivate[kernelConfig1D(LEN)](buffer, z_dvc, alpha_dvc, ss_dvc, st_dvc)
    cuda.synchronize()
    softmax_derivate[kernelConfig1D(LEN)](buffer, alpha_dvc, ss_dvc, st_dvc)
    cuda.synchronize()

    return buffer
