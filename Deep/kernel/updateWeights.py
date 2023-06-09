from numba import cuda
from ..lib.GPU import *
from timeit import default_timer as timer
from .vars import add
from ..transfer.loader import loadTo

def updateWeight(weights, eta, nabla_w, _):
    #print("updateWeight")
    LEN1, LEN2 = weights.shape

    t = timer()
    eta_dvc, nabla_w_dvc, w_dvc = loadTo(eta, nabla_w, weights, mode='GPU')
    t = timer() - t
    add(t)

    updateWeights[kernelConfig2D(LEN1, LEN2)](w_dvc, eta_dvc, nabla_w_dvc)
    cuda.synchronize()

    #weights = w_dvc.copy_to_host()
    return w_dvc