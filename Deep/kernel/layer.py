from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo
from colorama import Fore

def layer(x, w, b, buffer):
    LEN = w.shape[1]

    t = timer()
    x_dvc, w_dvc, b_dvc = loadTo(x, w, b, mode='GPU')
    t = timer() - t
    add(t)

    dotMatrix[kernelConfig1D(LEN)](buffer, x_dvc, w_dvc, b_dvc)
    cuda.synchronize()

    return buffer