from ..tensor import Tensor
from . import cpu

class LeakyReLU(object):
    def __init__(self, leak=0.01, gpu_mode=False):
        self._gpu = gpu_mode
        self._leak = leak
        self._type_layer = 'activation'
        self._cache: Tensor = None

    def type(self):
        return self._type_layer

    def send(self, in_data: Tensor):
        self._cache = in_data

        return cpu.leaky_relu(in_data=in_data, leak=self._leak)

    def learn(self, gradients: Tensor):
        return cpu.leaky_relu_derivate(
            in_data=self._cache,
            gradients=gradients,
            leak=self._leak
        )
