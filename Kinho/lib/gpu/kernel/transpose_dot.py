from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCKS = (1, 32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, const_A: DeviceNDArray, const_B: DeviceNDArray):
    batch_id, y_row, z_col = cuda.grid(3)
    
    sA = cuda.shared.array(shape=(THREADSPERBLOCKS[0], THREADSPERBLOCKS[1]), dtype=float64)
    sB = cuda.shared.array(shape=(THREADSPERBLOCKS[0], THREADSPERBLOCKS[2]), dtype=float64)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    if batch_id >= buffer.shape[0] or \
        y_row >= buffer.shape[1] or \
            z_col >= buffer.shape[2]:
                return None
    
    if ty == 0:
        sB[tx, tz] = const_B[batch_id, 0, z_col]
    
    if tz == 0:
        sA[tx, ty] = const_A[batch_id, 0, y_row]
    
    cuda.syncthreads()

    buffer[batch_id, y_row, z_col] = sA[tx, ty] * sB[tx, tz]

def transpose_dot(buffer: DeviceNDArray, const_A: DeviceNDArray, const_B: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCKS[0]),
         ceil(buffer.shape[1], THREADSPERBLOCKS[1]),
         ceil(buffer.shape[2], THREADSPERBLOCKS[2]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCKS)](buffer, const_A, const_B)
    cuda.synchronize()
