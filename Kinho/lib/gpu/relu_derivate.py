from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import relu_derivate as kernel_relu_derivate, list_multiplication

def relu_derivate(signals: DeviceNDArray, alphas: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates leaky relu derivate
    Args:
        signals (DeviceNDArray): [BATCH][1][N]
        alphas (DeviceNDArray): [BATCH][1][N]
        buffer (DeviceNDArray): [BATCH][1][N]

    Returns:
        DeviceNDArray: [BATCH][1][N]
    """
    kernel_relu_derivate(buffer=buffer, signals=signals)
    list_multiplication(buffer=buffer, const_A=buffer, const_B=alphas)
    
    return buffer
