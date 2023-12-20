from . import gpu, cpu
from numba import cuda
import numpy as np

class MSE:
    def __init__(self, inShape, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        self._typeLayer = 'cost'
        self._cache = None

        if self._gpu:
            arr = cuda.device_array((1,), dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)
            arr = cuda.device_array(inShape, dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)

    def type(self):
        return self._typeLayer

    def send(self, predict, target):
        self._cache = predict
        if self._gpu:
            return gpu.mse(predict=predict,
                           target=target,
                           buffer=self._outBuffer)
        else:
            return cpu.mse(predict=predict,
                           target=target)
    
    def learn(self, targets):
        if self._gpu:
            return gpu.mse_derivate(predicts=self._cache,
                                    targets=targets,
                                    buffer=self._inBuffer)
        else:
            return cpu.mse_derivate(predicts=self._cache,
                                    targets=targets)
