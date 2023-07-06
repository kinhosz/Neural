from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import memset_constant, mse as kernel_mse

def mse(predict: DeviceNDArray, target: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates mse error

    Args:
        predict (DeviceNDArray): [BATCH][1][N]
        target (DeviceNDArray): [BATCH][1][N]
        buffer (DeviceNDArray): [1]

    Returns:
        DeviceNDArray: buffer
    """
    
    memset_constant(buffer)
    kernel_mse(buffer=buffer, predict=predict, target=target)
    
    return buffer
