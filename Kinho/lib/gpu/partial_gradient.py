from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import partial_gradient as partial

def partial_gradient(weight: DeviceNDArray, eta: DeviceNDArray, gradient: DeviceNDArray) -> DeviceNDArray:
    """update: weight - eta * gradient

    Args:
        weight (DeviceNDArray): [N][M]
        eta (DeviceNDArray): [1]
        gradient (DeviceNDArray): [N][M]

    Returns:
        DeviceNDArray: [N][M]
    """
    
    partial(buffer=weight, const_A=weight, const_B=gradient, eta=eta)
    
    return weight
