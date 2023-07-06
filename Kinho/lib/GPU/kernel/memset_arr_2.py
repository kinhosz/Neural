from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCK = (16, 64)

@cuda.jit
def perform(buffer: DeviceNDArray):
    row, col = cuda.grid(2)
    
    if row >= buffer.shape[0] or \
        col >= buffer.shape[1]:
            return None
    
    buffer[row, col] = float64(0.0)

def memset_arr_2(buffer: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[1], THREADSPERBLOCK[1]))
    )

    perform[(blockspergrid, THREADSPERBLOCK)](buffer)
    cuda.synchronize()
