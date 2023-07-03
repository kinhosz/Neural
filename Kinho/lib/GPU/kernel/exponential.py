from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config
import math

THREADSPERBLOCK = (64, 16)

@cuda.jit
def perform(buffer: DeviceNDArray, signals: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    buffer[batch_id, 0, y_col] = math.exp(signals[batch_id, 0, y_col])

def exponential(buffer: DeviceNDArray, signals: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
        ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
    cuda.synchronize()
