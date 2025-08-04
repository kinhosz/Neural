from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from .utils import ceil, grid_config

import math

THREADSPERBLOCK = (128, 8)

@cuda.jit
def perform(buffer: DeviceNDArray, predict: DeviceNDArray, target: DeviceNDArray):
    batch_id, x = cuda.grid(2)

    if batch_id >= target.shape[0] or x >= target.shape[2]:
        return None

    t_x = target[batch_id][0][x]
    p_x = predict[batch_id][0][x]

    res = (t_x * math.log(p_x)) + ((1.0 - t_x) * math.log(1.0 - p_x))
    res = (-1.0 / (predict.shape[0] + predict.shape[2])) * res

    cuda.atomic.add(buffer, 0, res)

def bce(buffer: DeviceNDArray, predict: DeviceNDArray, target: DeviceNDArray):
    blockspergrid = grid_config(
        (ceil(target.shape[0], THREADSPERBLOCK[0]),
         ceil(target.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, predict, target)
    cuda.synchronize()
