from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .utils import grid_config, ceil
from numba import cuda

THREADSPERBLOCK = (32, 32)

@cuda.jit
def perform(buffer: DeviceNDArray, const_z: DeviceNDArray):
    batch_id, x = cuda.grid(2)
    
    if batch_id >= buffer.shape[0] or x >= const_z.shape[2]:
        return None

    ALPHA = 0.01
    r = const_z[batch_id][0][x]
    if r < 0.0:
        r = r * ALPHA

    buffer[batch_id, 0, x] = r

def relu(buffer: DeviceNDArray, signals: DeviceNDArray):
    """ leaky relu

    Args:
        signals (DeviceNDArray): [batch][1][N]
        buffer (DeviceNDArray): [batch][1][N]
    """
    blockspergrid = grid_config(
        (ceil(signals.shape[0], THREADSPERBLOCK[0]), 
        ceil(signals.shape[2], THREADSPERBLOCK[1]))
    )
    
    perform[(blockspergrid, THREADSPERBLOCK)](buffer, signals)
    cuda.synchronize()
