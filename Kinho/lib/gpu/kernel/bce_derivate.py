from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (128, 8)

@cuda.jit
def perform(buffer: DeviceNDArray, predicts: DeviceNDArray, targets: DeviceNDArray):
    batch_id, x = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or x >= buffer.shape[2]:
        return None
    
    N = buffer.shape[0]
    t_x = targets[batch_id][0][x]
    p_x = predicts[batch_id][0][x]

    res = (p_x - t_x) / (p_x * (1.0 - p_x))
    res = res / N

    buffer[batch_id][0][x] = res

def bce_derivate(buffer: DeviceNDArray, predicts: DeviceNDArray, targets: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, predicts, targets)
    cuda.synchronize()
