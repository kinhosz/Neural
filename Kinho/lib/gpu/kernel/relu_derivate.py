from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, signals: DeviceNDArray):
    batch_id, x = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or x >= buffer.shape[2]:
        return None
    
    ALPHA = 0.01
    r = 1.0
    if signals[batch_id][0][x] < 0.0:
        r = ALPHA

    buffer[batch_id, 0, x] = r

def relu_derivate(buffer: DeviceNDArray, signals: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
