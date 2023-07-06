from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (64, 16)

@cuda.jit
def perform(buffer: DeviceNDArray, signals: DeviceNDArray, const_values: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= signals.shape[0] or \
        y_col >= signals.shape[2]:
            return None
    
    buffer[batch_id, 0, y_col] = signals[batch_id, 0, y_col] / const_values[batch_id, 0]

def divide_by_const(buffer: DeviceNDArray, signals: DeviceNDArray, const_values: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(signals.shape[0], THREADSPERBLOCK[0]),
        ceil(signals.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals, const_values)
    cuda.synchronize()
