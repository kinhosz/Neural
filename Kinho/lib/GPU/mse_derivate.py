from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import mse_derivate as kernel_mse_derivate

def mse_derivate(predicts: DeviceNDArray, targets: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates: error[i] = (2/N) * (predict[i] - target[i])

    Args:
        predicts (DeviceNDArray): [Batch][1][N]
        targets (DeviceNDArray): [Batch][1][N]
        buffer (DeviceNDArray): [Batch][1][N]

    Returns:
        DeviceNDArray: [Batch][1][N]
    """
    kernel_mse_derivate(buffer=buffer, predicts=predicts, targets=targets)
    
    return buffer
