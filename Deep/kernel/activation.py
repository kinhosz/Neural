from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo
def activation(z, buffer):
    #print("activation")
    LEN = z.shape[1]

    t = timer()
    z_dvc, = loadTo(z, mode='GPU')
    t = timer() - t
    add(t)

    sigmoid2[kernelConfig1D(LEN)](buffer, z_dvc)
    cuda.synchronize()

    #arr = arr_dvc.copy_to_host()
    return buffer