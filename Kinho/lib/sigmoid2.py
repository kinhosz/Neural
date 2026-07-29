from . import gpu, cpu
from numba import cuda
import numpy as np

class Sigmoid2:
    def __init__(self, in_shape, out_shape, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        self._typeLayer = 'activation'
        self._cache = None

        if self._gpu:
            arr = cuda.device_array(in_shape, dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array(out_shape, dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)
        
    def type(self):
        return self._typeLayer

    def send(self, in_data):
        self._cache = in_data

        if self._gpu:
            return gpu.sigmoid2(signals=in_data, buffer=self._outBuffer)
        else:
            return cpu.sigmoid2(signals=in_data)

    def learn(self, gradients):
        if self._gpu:
            return gpu.sigmoid2_derivate(signals=self._cache,
                                        alphas=gradients,
                                        buffer=self._inBuffer)
        else:
            return cpu.sigmoid2_derivate(signals=self._cache,
                                         alphas=gradients)
