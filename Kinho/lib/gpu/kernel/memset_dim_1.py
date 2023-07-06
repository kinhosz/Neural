from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (256, 4)

@cuda.jit
def perform(buffer: DeviceNDArray):
    batch_id, idx = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        idx >= buffer.shape[1]:
            return None
    
    buffer[batch_id, idx] = float(0.0)

def memset_dim_1(buffer: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
        ceil(buffer.shape[1], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer)
    cuda.synchronize()
