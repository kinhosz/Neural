from ..tensor import Tensor
from . import cpu, gpu

class MaxPooling(object):
    def __init__(self, gpu_mode=False):
        self._gpu = gpu_mode
        self._type_layer = 'pooling'
        self._cache: Tensor = None

    def type(self):
        return self._type_layer

    def send(self, in_data: Tensor):
        self._cache = in_data

        if self._gpu:
            return gpu.max_pooling()
        else:
            return cpu.max_pooling(in_data=in_data)

    def learn(self, gradients: Tensor):
        if self._gpu:
            return gpu.max_pooling_derivate()
        else:
            return cpu.max_pooling_derivate(
                in_data=self._cache,
                gradients=gradients
            )
