from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .utils import grid_config, ceil
from numba import cuda

THREADSPERBLOCK = (64, 16)

@cuda.jit
def perform(buffer: DeviceNDArray, exps: DeviceNDArray, sum_exps: DeviceNDArray):
    batch_id, y_col = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or \
        y_col >= buffer.shape[2]:
            return None
    
    exp_value = buffer[batch_id, 0, y_col]
    exp_sum_value = sum_exps[batch_id, 0]
    
    res_value = (exp_value * exp_sum_value) - (exp_value * exp_value)
    res_value /= (exp_sum_value * exp_sum_value)
    
    buffer[batch_id, 0, y_col] = res_value

def softmax_derivate(buffer: DeviceNDArray, exps: DeviceNDArray, sum_exps: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(buffer.shape[0], THREADSPERBLOCK[0]),
         ceil(buffer.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, exps, sum_exps)
    cuda.synchronize()
