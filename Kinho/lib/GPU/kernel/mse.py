from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (1024,)

@cuda.jit
def perform(buffer: DeviceNDArray, predict: DeviceNDArray, target: DeviceNDArray):
    x = cuda.grid(1)
    
    if x >= target.shape[2]:
        return None
    
    diff = predict[0, 0, x] - target[0, 0, x]
    diff_square_divide_N = (diff * diff)/target.shape[2]
    
    cuda.atomic.add(buffer, 0, diff_square_divide_N)

def mse(buffer: DeviceNDArray, predict: DeviceNDArray, target: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(target.shape[2], THREADSPERBLOCK[0]),)
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, predict, target)
    cuda.synchronize()
