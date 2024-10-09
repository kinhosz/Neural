from ..tensor import Tensor
from . import cpu, gpu

class ConvLayer(object):
    def __init__(self, weight: Tensor, biase: Tensor, gpu_mode=False) -> None:
        self._gpu = gpu_mode
        self._weight = weight
        self._biase = biase
        self._cache: Tensor = None

    def send(self, in_data: Tensor):
        self._cache = in_data

        if self._gpu:
            return gpu.convolution()
        else:
            return cpu.convolution(
                in_data=in_data,
                weight=self._weight,
                biase=self._biase
            )

    def learn(self, gradients):
        if self._gpu:
            return gpu.convolution_derivate()
        else:
            return cpu.convolution_derivate()
