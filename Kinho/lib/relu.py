from . import gpu, cpu
from numba import cuda
import numpy as np

class ReLU:
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

    def send(self, signals):
        self._cache = signals

        if self._gpu:
            return gpu.relu(signals=signals, buffer=self._outBuffer)
        else:
            return cpu.relu(signals=signals)

    def learn(self, alphas):
        if self._gpu:
            return gpu.relu_derivate(signals=self._cache,
                                        alphas=alphas,
                                        buffer=self._inBuffer)
        else:
            return cpu.relu_derivate(signals=self._cache,
                                         alphas=alphas)
