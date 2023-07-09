from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCK = (32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, const_host: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    sH = cuda.shared.array(shape=(THREADSPERBLOCK[1],), dtype=float64)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    if tx == 0:
        sH[ty] = const_host[0, y_col]
    
    if batch_id >= buffer.shape[0] or \
        y_col >= const_host.shape[1]:
            return None
    
    cuda.syncthreads()
    
    buffer[batch_id, 0, y_col] = sH[ty]

def copy(buffer: DeviceNDArray, const_host: DeviceNDArray):
    """Perform: buffer[batch_id] = const_host

    Args:
        buffer (DeviceNDArray): [batch][1][N]
        const_host (DeviceNDArray): [1][N]
    """

    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]), ceil(const_host.shape[1], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, const_host)
    cuda.synchronize()
