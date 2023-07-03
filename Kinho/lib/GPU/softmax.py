from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import exponential, sum_list, divide_by_const, memset_dim_1

def softmax(signals: DeviceNDArray, extra: DeviceNDArray, buffer: DeviceNDArray):
    """e^(z)/sum{e^z}

    Args:
        signals (DeviceNDArray): [batch][1][N]
        extra (DeviceNDArray): [batch][1]
        buffer (DeviceNDArray): [batch][1][N]
    
    returns:
        buffer
    """
    exponential(buffer=buffer, signals=signals)
    memset_dim_1(buffer=extra)
    sum_list(buffer=extra, signals=buffer)
    divide_by_const(buffer=buffer, signals=buffer, const_values=extra)
    
    return buffer
