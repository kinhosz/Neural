from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import average_batch, memset_arr_2

def stochastic_gradient_descent(gradients: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """sgd for gradients

    Args:
        gradients (DeviceNDArray): [BATCH][N][M]
        buffer (DeviceNDArray): [N][M]

    Returns:
        DeviceNDArray: buffer
    """
    memset_arr_2(buffer=buffer)
    average_batch(buffer=buffer, gradients=gradients)
    
    return buffer
