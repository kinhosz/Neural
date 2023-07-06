from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import sigmoid_0_1

def sigmoid2(signals: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates the sigmoid in (-1, 1)

    Args:
        signals (DeviceNDArray): [batch][1][N]
        buffer (DeviceNDArray): [batch][1][N]

    Returns:
        (DeviceNDArray): [batch][1][N]
    """
    sigmoid_0_1(buffer=buffer, signals=signals)
    
    return buffer
