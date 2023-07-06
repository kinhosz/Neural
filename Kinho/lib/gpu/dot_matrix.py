from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import matrix_multiplication, copy

def dot_matrix(
    signals: DeviceNDArray,
    weight: DeviceNDArray,
    bias: DeviceNDArray,
    buffer: DeviceNDArray,
):
    """buffer[batch_id] = signails[batch_id] * weight + bias

    Args:
        signals (DeviceNDArray): [batch_id][1][N]
        weight (DeviceNDArray): [N][M]
        bias (DeviceNDArray): [1][M]
        buffer (DeviceNDArray): [batch_id][1][M]

    Returns:
        buffer
    """
    copy(buffer, bias)
    matrix_multiplication(buffer=buffer, const_a=signals, const_b=weight)

    return buffer
