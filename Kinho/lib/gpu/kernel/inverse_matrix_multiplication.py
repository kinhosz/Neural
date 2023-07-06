from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCK = (1, 32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, const_matrix: DeviceNDArray, alphas: DeviceNDArray):
    batch_id, y_row, z_col = cuda.grid(3)
    
    sMatrix = cuda.shared.array(shape=(THREADSPERBLOCK[1], THREADSPERBLOCK[2]), dtype=float64)
    sAlpha = cuda.shared.array(shape=(THREADSPERBLOCK[0], THREADSPERBLOCK[1]), dtype=float64)
    tmp = cuda.shared.array(shape=THREADSPERBLOCK, dtype=float64)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    tmp[tx, ty, tz] = float64(0.0)
    
    if batch_id >= buffer.shape[0] or \
        y_row >= const_matrix.shape[1] or \
            z_col >= const_matrix.shape[0]: # inverse
                return None

    if tx == 0:
        sMatrix[ty, tz] = const_matrix[z_col, y_row] # inverse
    
    if tz == 0:
        sAlpha[tx, ty] = alphas[batch_id, 0, y_row]
    
    cuda.syncthreads()
    
    tmp[tx, ty, tz] = sAlpha[tx, ty] * sMatrix[ty, tz]
    
    cuda.syncthreads()
    
    if ty != 0:
        return None
    
    for i in range(1, cuda.blockDim.y):
        tmp[tx, 0, tz] += tmp[tx, i, tz]
    
    cuda.atomic.add(buffer, (batch_id, 0, z_col), tmp[tx, 0, tz])

def inverse_matrix_multiplication(buffer: DeviceNDArray, const_matrix: DeviceNDArray, alphas: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(const_matrix.shape[1], THREADSPERBLOCK[1]), # inverse shape
         ceil(const_matrix.shape[0], THREADSPERBLOCK[2]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, const_matrix, alphas)
    cuda.synchronize()
