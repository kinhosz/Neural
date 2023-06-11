from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def dlayer(w, alpha, buffer):
    LEN = w.shape[0]
    LEN2 = w.shape[1]

    t = timer()
    w_dvc, alpha_dvc = loadTo(w, alpha, mode='GPU')
    t = timer() - t
    add(t)

    dotMatrix_derivate[kernelConfig3D(1, LEN, LEN2)](buffer, w_dvc, alpha_dvc)
    cuda.synchronize()

    return buffer