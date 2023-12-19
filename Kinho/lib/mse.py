from . import gpu, cpu
from numba import cuda
import numpy as np

class MSE:
    def __init__(self, outShape, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        self._typeLayer = 'cost'

        if self._gpu:
            arr = cuda.device_array((1,), dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array(outShape, dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)

    def type(self):
        return self._typeLayer

    def send(self, predict, target):
        if self._gpu:
            return gpu.mse(predict=predict,
                           target=target,
                           buffer=self._inBuffer)
        else:
            return cpu.mse(predict=predict,
                           target=target)
    
    def learn(self, predicts, targets):
        if self._gpu:
            return gpu.mse_derivate(predicts=predicts,
                                    targets=targets,
                                    buffer=self._outBuffer)
        else:
            return cpu.mse_derivate(predicts=predicts,
                                    targets=targets)
