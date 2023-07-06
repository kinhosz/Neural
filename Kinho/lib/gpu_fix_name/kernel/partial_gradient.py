from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

THREADSPERBLOCK = (8, 128)

@cuda.jit
def perform(buffer: DeviceNDArray, const_A: DeviceNDArray, const_B: DeviceNDArray, eta: DeviceNDArray):
    row, col = cuda.grid(2)
    
    if row >= buffer.shape[0] or \
        col >= buffer.shape[1]:
            return None
    
    buffer[row, col] = const_A[row, col] - eta[0] * const_B[row, col]

def partial_gradient(buffer: DeviceNDArray, const_A: DeviceNDArray, const_B: DeviceNDArray, eta: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[1], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, const_A, const_B, eta)
    cuda.synchronize()
