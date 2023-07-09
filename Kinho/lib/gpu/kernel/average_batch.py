from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda, float64
from .utils import ceil, grid_config

THREADSPERBLOCK = (64, 1, 16)

@cuda.jit
def perform(buffer: DeviceNDArray, gradients: DeviceNDArray):
    batch_id, y_row, z_col = cuda.grid(3)
    
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z
    
    tmp = cuda.shared.array(shape=THREADSPERBLOCK, dtype=float64)
    tmp[tx, ty, tz] = float64(0.0)
    
    if batch_id >= gradients.shape[0] or \
        y_row >= gradients.shape[1] or \
            z_col >= gradients.shape[2]:
                return None
    
    cuda.syncthreads()
    
    tmp[tx, ty, tz] = gradients[batch_id, y_row, z_col]
    
    cuda.syncthreads()
    
    if tx != 0:
        return None
    
    for k in range(1, cuda.blockDim.x):
        tmp[0, ty, tz] += tmp[k, ty, tz]
    
    cuda.atomic.add(buffer, (y_row, z_col), tmp[0, ty, tz])

def average_batch(buffer: DeviceNDArray, gradients: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(gradients.shape[0], THREADSPERBLOCK[0]),
         ceil(gradients.shape[1], THREADSPERBLOCK[1]),
         ceil(gradients.shape[2], THREADSPERBLOCK[2]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, gradients)
    cuda.synchronize()
