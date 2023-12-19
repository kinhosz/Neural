from . import gpu, cpu
from numba import cuda
import numpy as np

class Softmax:
    def __init__(self, in_shape, out_shape, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        self._typeLayer = 'selector'

        if self._gpu:
            arr = cuda.device_array(in_shape, dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array(out_shape, dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)
            arr = cuda.device_array((in_shape[0], in_shape[1]), dtype=np.float64)
            self._extra = cuda.to_device(arr)

    def type(self):
        return self._typeLayer

    def send(self, signals):
        if self._gpu:
            return gpu.softmax(signals=signals,
                               extra=self._extra,
                               buffer=self._inBuffer)
        else:
            return cpu.softmax(signals=signals)
    
    def learn(self, signals, alphas):
        if self._gpu:
            return gpu.softmax_derivate(signals=signals,
                                        alphas=alphas,
                                        extra=self._extra,
                                        buffer=self._outBuffer)
        else:
            return cpu.softmax_derivate(signals=signals,
                                         alphas=alphas)
