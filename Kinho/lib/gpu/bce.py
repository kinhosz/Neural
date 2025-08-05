from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import memset_constant, bce as kernel_bce

def bce(predict: DeviceNDArray, target: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates bce error

    Args:
        predict (DeviceNDArray): [BATCH][1][N]
        target (DeviceNDArray): [BATCH][1][N]
        buffer (DeviceNDArray): [1]

    Returns:
        DeviceNDArray: buffer
    """
    
    memset_constant(buffer)
    kernel_bce(buffer=buffer, predict=predict, target=target)
    
    return buffer
