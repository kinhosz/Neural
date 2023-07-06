from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (64, 16)

@cuda.jit
def perform(buffer: DeviceNDArray, signals: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= signals.shape[0] or \
        y_col >= signals.shape[2]:
            return None
    
    cuda.atomic.add(buffer, (batch_id, 0), signals[batch_id, 0, y_col])

def sum_list(buffer: DeviceNDArray, signals: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(signals.shape[0], THREADSPERBLOCK[0]),
        ceil(signals.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
    cuda.synchronize()
