from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import sigmoid_0_1_derivate, list_multiplication

def sigmoid2_derivate(signals: DeviceNDArray, alphas: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates sigmoid_0_1 derivate

    Args:
        signals (DeviceNDArray): [BATCH][1][N]
        alphas (DeviceNDArray): [BATCH][1][N]
        buffer (DeviceNDArray): [BATCH][1][N]

    Returns:
        DeviceNDArray: [BATCH][1][N]
    """
    sigmoid_0_1_derivate(buffer=buffer, signals=signals)
    list_multiplication(buffer=buffer, const_A=buffer, const_B=alphas)
    
    return buffer
