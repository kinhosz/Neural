from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo
from colorama import Fore

def layer(x, w, b, buffer):
    LEN = w.shape[0]
    LEN2 = w.shape[1]

    t = timer()
    x_dvc, w_dvc, b_dvc = loadTo(x, w, b, mode='GPU')
    t = timer() - t
    add(t)

    copy[kernelConfig1D(buffer.shape[1])](buffer, b_dvc)
    cuda.synchronize()
    dotMatrix[kernelConfig2D(LEN, LEN2, shape=(4, 256))](buffer, x_dvc, w_dvc)
    cuda.synchronize()

    return buffer
