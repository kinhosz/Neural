from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import multiplication, copy

def dot_matrix(
    signals: DeviceNDArray,
    weight: DeviceNDArray,
    bias: DeviceNDArray,
    buffer: DeviceNDArray,
):
    copy(buffer, bias)
    multiplication(buffer=buffer, const_a=signals, const_b=weight)

    return buffer
