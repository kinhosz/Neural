from . import gpu, cpu
from numba import cuda
import numpy as np

class DotMatrix:
    def __init__(self, batchsize, weight, biase, eta, gpuMode=False):
        self._gpu = gpuMode
        self._eta = eta
        self._inBuffer = None
        self._outBuffer = None
        self._transposeBuffer = None
        self._updateWeightBuffer = None
        self._updateBiasBuffer = None
        self._batchsize = batchsize
        self._weight = weight
        self._biase = biase
        self._typeLayer = 'neurons'
        self._cache = None

        if self._gpu:
            self._weight = cuda.to_device(weight)
            self._biase = cuda.to_device(biase)

            arr = cuda.device_array((self._batchsize, ) + (1, weight.shape[1]), dtype=np.float64)
            self._inBuffer = cuda.to_device(arr)
            arr = cuda.device_array((self._batchsize, ) + (1, weight.shape[0]), dtype=np.float64)
            self._outBuffer = cuda.to_device(arr)
            
            arr = cuda.device_array((batchsize, ) + weight.shape, dtype=np.float64)
            self._transposeBuffer = cuda.to_device(arr)
            
            arr = cuda.device_array(weight.shape, dtype=np.float64)
            self._updateWeightBuffer = cuda.to_device(arr)
            
            arr = cuda.device_array(biase.shape, dtype=np.float64)
            self._updateBiasBuffer = cuda.to_device(arr)

    def type(self):
        return self._typeLayer

    def send(self, in_data):
        self._cache = in_data
        if self._gpu:
            return gpu.dot_matrix(signals=in_data,
                                  weight=self._weight,
                                  bias=self._biase,
                                  buffer=self._inBuffer)
        else:
            return cpu.dot_matrix(signals=in_data,
                                  weight=self._weight,
                                  bias=self._biase)

    def learn(self, gradients):
        nabla_w = self._transpose(self._cache, gradients)
        nabla_b = gradients
        response = None

        if self._gpu:
            response = gpu.dot_matrix_derivate(const_matrix=self._weight, 
                                                alphas=gradients, 
                                                buffer=self._outBuffer)
        else:
            response = cpu.dot_matrix_derivate(const_matrix=self._weight, 
                                                alphas=gradients)

        self._updateWeight(nabla_w)
        self._updateBias(nabla_b)
        
        return response
    
    def weight(self):
        return self._weight.copy_to_host() if self._gpu else self._weight
    
    def bias(self):
        return self._biase.copy_to_host() if self._gpu else self._biase
    
    def _transpose(self, signals, alphas):
        if self._gpu:
            return gpu.transpose(signals=signals,
                                alphas=alphas,
                                buffer=self._transposeBuffer)
        else:
            return cpu.transpose(signals=signals,
                                alphas=alphas)

    def _updateWeight(self, nabla):
        sgd = self._stochastic_gradient_descent(nabla, self._updateWeightBuffer)
        
        if self._gpu:
            self._weight = gpu.partial_gradient(weight=self._weight,
                                                eta=self._eta,
                                                gradient=sgd)
        else:
            self._weight = cpu.partial_gradient(weight=self._weight,
                                                eta=self._eta,
                                                gradient=sgd)
    
    def _updateBias(self, nabla):
        sgd = self._stochastic_gradient_descent(nabla, self._updateBiasBuffer)
        
        if self._gpu:
            self._biase = gpu.partial_gradient(weight=self._biase,
                                                eta=self._eta,
                                                gradient=sgd)
        else:
            self._biase = cpu.partial_gradient(weight=self._biase,
                                                eta=self._eta,
                                                gradient=sgd)
    
    def _stochastic_gradient_descent(self, gradients, buffer):
        if self._gpu:
            return gpu.stochastic_gradient_descent(gradients=gradients,
                                                   buffer=buffer)
        else:
            return cpu.stochastic_gradient_descent(gradients=gradients)
