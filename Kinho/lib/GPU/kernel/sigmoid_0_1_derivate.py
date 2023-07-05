from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config
import math

THREADSPERBLOCK = (8, 128)

@cuda.jit
def perform(buffer: DeviceNDArray, signals: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    exp_value = math.exp(-signals[batch_id, 0, y_col])
    exp_plus_one_square = (1.0 + exp_value) * (1.0 + exp_value)
    
    buffer[batch_id, 0, y_col] = (2.0 * exp_value)/exp_plus_one_square

def sigmoid_0_1_derivate(buffer: DeviceNDArray, signals: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
