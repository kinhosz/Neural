from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import exponential, sum_list, memset_dim_1, list_multiplication
from .kernel import softmax_derivate as soft_derivate

def softmax_derivate(signals: DeviceNDArray, alphas: DeviceNDArray, extra: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    exponential(buffer=buffer, signals=signals)
    memset_dim_1(buffer=extra)
    sum_list(buffer=extra, signals=buffer)
    soft_derivate(buffer=buffer, exps=buffer, sum_exps=extra)
    list_multiplication(buffer=buffer, const_A=buffer, const_B=alphas)
    
    return buffer
