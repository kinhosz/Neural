from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import grid_config, ceil

THREADSPERBLOCK = (32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    buffer[batch_id, 0, y_col] = float64(0.0)

def memset_arr_1(buffer: DeviceNDArray):
    """reset arr with 0s

    Args:
        buffer (DeviceNDArray): [BATCH][1][N]
    """
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer)
    cuda.synchronize()
