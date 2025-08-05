from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import bce_derivate as kernel_bce_derivate

def bce_derivate(predicts: DeviceNDArray, targets: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """evaluates: bce derivate, check cpu version

    Args:
        predicts (DeviceNDArray): [Batch][1][N]
        targets (DeviceNDArray): [Batch][1][N]
        buffer (DeviceNDArray): [Batch][1][N]

    Returns:
        DeviceNDArray: [Batch][1][N]
    """
    kernel_bce_derivate(buffer=buffer, predicts=predicts, targets=targets)
    
    return buffer
