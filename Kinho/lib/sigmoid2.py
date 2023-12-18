from . import gpu, cpu
from numba import cuda
import numpy as np

class Sigmoid2:
    def __init__(self, in_shape, out_shape, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        if self._gpu:
            arr = cuda.device_array(in_shape, dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array(out_shape, dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)

    def send(self, signals):
        if self._gpu:
            return gpu.sigmoid2(signals=signals, buffer=self._inBuffer)
        else:
            return cpu.sigmoid2(signals=signals)
    
    def learn(self, signals, alphas):
        if self._gpu:
            return gpu.sigmoid2_derivate(signals=signals,
                                        alphas=alphas,
                                        buffer=self._outBuffer)
        else:
            return cpu.sigmoid2_derivate(signals=signals,
                                         alphas=alphas)
