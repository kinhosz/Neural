from numba import cuda
from ..transfer.loader import loadTo

# implement batch
@DeprecationWarning
def layer(x, w, b, buffer):
    LEN = w.shape[0]
    LEN2 = w.shape[1]

    x_dvc, w_dvc, b_dvc = loadTo(x, w, b, mode='GPU')

    copy[kernelConfig1D(buffer.shape[1])](buffer, b_dvc)
    cuda.synchronize()
    dotMatrix[kernelConfig2D(LEN, LEN2, shape=(4, 256))](buffer, x_dvc, w_dvc)
    cuda.synchronize()

    return buffer
