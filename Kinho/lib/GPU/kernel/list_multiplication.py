from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (8, 128)

@cuda.jit
def perform(buffer: DeviceNDArray, A: DeviceNDArray, B: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)

    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    buffer[batch_id, 0, y_col] = A[batch_id, 0, y_col] * B[batch_id, 0, y_col]

def list_multiplication(buffer: DeviceNDArray, const_A: DeviceNDArray, const_B: DeviceNDArray):
    """buffer[i] = const_A[i] * const_B[i]

    Args:
        buffer (DeviceNDArray): [BATCH][1][N]
        const_A (DeviceNDArray): [BATCH][1][N]
        const_B (DeviceNDArray): [BATCH][1][N]
    """
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, const_A, const_B)
    cuda.synchronize()
    