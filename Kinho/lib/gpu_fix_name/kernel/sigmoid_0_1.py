from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .utils import grid_config, ceil
from numba import cuda
import math

THREADSPERBLOCK = (8, 128)

@cuda.jit
def perform(buffer: DeviceNDArray, const_z: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= const_z.shape[2]:
            return None
    
    buffer[batch_id, 0, y_col] = 2.0 * (1.0 / (1.0 + math.exp(-const_z[batch_id, 0, y_col]))) - 1.0

def sigmoid_0_1(buffer: DeviceNDArray, signals: DeviceNDArray):
    """sigmoid with media: 0, std-dvt: 1

    Args:
        signals (DeviceNDArray): [batch][1][N]
        buffer (DeviceNDArray): [batch][1][N]
    """
    blockspergrid = grid_config(
        (ceil(signals.shape[0], THREADSPERBLOCK[0]), 
        ceil(signals.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
    cuda.synchronize()
