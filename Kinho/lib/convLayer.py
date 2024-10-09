from ..tensor import Tensor
from . import cpu, gpu

class ConvLayer(object):
    def __init__(self, weight: Tensor, biase: Tensor, eta=0.1, gpu_mode=False) -> None:
        self._gpu = gpu_mode
        self._eta = eta
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

    def learn(self, gradients: Tensor):
        if self._gpu:
            response = gpu.convolution_derivate()
        else:
            response = cpu.convolution_derivate(
                weight=self._weight,
                gradients=gradients
            )

        self._update_layer(gradients)

        return response

    def _update_layer(self, gradients: Tensor):
        if self._gpu:
            pass
        else:
            biase_gradients = cpu.get_biase_gradients(gradients)
            cpu.apply_biase_gradients(self._biase, biase_gradients, self._eta)

            weight_gradients = cpu.get_weight_gradients(self._cache, gradients)
            cpu.apply_weight_gradients(self._weight, weight_gradients, self._eta)
