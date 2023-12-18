from . import gpu, cpu
from numba import cuda
import numpy as np

class DotMatrix:
    def __init__(self, batchsize, weight, biase, gpuMode=False):
        self._gpu = gpuMode
        self._inBuffer = None
        self._outBuffer = None
        self._batchsize = batchsize
        self._weight = weight
        self._biase = biase

        if self._gpu:
            self._weight = cuda.to_device(weight)
            self._biase = cuda.to_device(biase)

            arr = cuda.device_array((self._batchsize, ) + (1, weight.shape[1]), dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array((self._batchsize, ) + (1, weight.shape[0]), dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)

    def send(self, signals):
        if self._gpu:
            return gpu.dot_matrix(signals=signals,
                                  weight=self._weight,
                                  bias=self._biase,
                                  buffer=self._inBuffer)
        else:
            return cpu.dot_matrix(signals=signals,
                                  weight=self._weight,
                                  bias=self._biase)
    
    def learn(self, alphas):
        if self._gpu:
            return gpu.dot_matrix_derivate(const_matrix=self._weight, 
                                           alphas=alphas, 
                                           buffer=self._outBuffer)
        else:
            return cpu.dot_matrix_derivate(const_matrix=self._weight, 
                                           alphas=alphas)
