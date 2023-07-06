from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (128, 8)

@cuda.jit
def perform(buffer: DeviceNDArray, predicts: DeviceNDArray, targets: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    N = buffer.shape[2]
    
    buffer[batch_id, 0, y_col] = (2.0 / N) * (predicts[batch_id, 0, y_col] - targets[batch_id, 0, y_col])

def mse_derivate(buffer: DeviceNDArray, predicts: DeviceNDArray, targets: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, predicts, targets)
    cuda.synchronize()
