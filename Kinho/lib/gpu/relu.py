from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import relu as kernel_relu

def relu(signals: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates the leaky relu

    Args:
        signals (DeviceNDArray): [batch][1][N]
        buffer (DeviceNDArray): [batch][1][N]

    Returns:
        (DeviceNDArray): [batch][1][N]
    """
    kernel_relu(buffer=buffer, signals=signals)
    
    return buffer
