from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import grid_config, ceil

# (batch, A.y, B.y)
THREADSPERBLOCK = (1, 32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, const_a: DeviceNDArray, const_b: DeviceNDArray):
    batch_id, y_row, z_col = cuda.grid(3)

    tmp = cuda.shared.array(shape=THREADSPERBLOCK, dtype=float64)
    sA = cuda.shared.array(shape=(THREADSPERBLOCK[0], THREADSPERBLOCK[1]), dtype=float64)
    sB = cuda.shared.array(shape=(THREADSPERBLOCK[1], THREADSPERBLOCK[2]), dtype=float64)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    tmp[tx, ty, tz] = float64(0.0)
    
    if batch_id >= const_a.shape[0] or \
        y_row >= const_b.shape[0] or \
            z_col >= const_b.shape[1]:
                return None
    if tz == 0:
        sA[tx, ty] = const_a[batch_id, 0, y_row]
    
    if tx == 0:
        sB[ty, tz] = const_b[y_row, z_col]
    
    cuda.syncthreads()
    
    tmp[tx, ty, tz] = sA[tx, ty] * sB[ty, tz]
    
    cuda.syncthreads()
    
    if ty != 0:
        return None
    
    for i in range(1, cuda.blockDim.y):
        tmp[tx, 0, tz] += tmp[tx, i, tz]
    
    cuda.atomic.add(buffer, (batch_id, 0, z_col), tmp[tx, 0, tz])

def matrix_multiplication(buffer: DeviceNDArray, const_a: DeviceNDArray, const_b: DeviceNDArray) -> None:
    """ Perform buffer[batch_id] = const_a[batch_id] * const_b

    Args:
        buffer (DeviceNDArray): [batch][1][M]
        const_a (DeviceNDArray): [batch][1][N]
        const_b (DeviceNDArray): [N][M]
    """
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(const_b.shape[0], THREADSPERBLOCK[1]),
         ceil(const_b.shape[1], THREADSPERBLOCK[2]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, const_a, const_b)
    cuda.synchronize()
