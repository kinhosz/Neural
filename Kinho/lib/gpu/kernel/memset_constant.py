from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCK = (1024,)

@cuda.jit
def perform(buffer):
    x = cuda.grid(1)
    
    if x >= buffer.shape[0]:
        return None
    
    buffer[x] = float64(0.0)

def memset_constant(buffer: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(1, THREADSPERBLOCK[0]),)
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer)
    cuda.synchronize()
