from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import transpose_dot

def transpose(signals: DeviceNDArray, alphas: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates the gradient for a matrix_multiplication
    using signals as input and alphas as error

    Args:
        signals (DeviceNDArray): [BATCH][1][N]
        alphas (DeviceNDArray): [BATCH][1][M]
        buffer (DeviceNDArray): [BATCH][N][M]

    Returns:
        DeviceNDArray: buffer
    """
    transpose_dot(buffer=buffer, const_A=signals, const_B=alphas)
    
    return buffer
