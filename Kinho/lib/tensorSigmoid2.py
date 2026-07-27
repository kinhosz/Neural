from ..tensor import Tensor
from . import cpu

class TensorSigmoid2(object):
    """Sigmoid2 activation for the conv backbone, where layers exchange
    Tensor objects (K, N, M) instead of the dense head's batched ndarrays."""

    def __init__(self):
        self._type_layer = 'activation'
        self._cache: Tensor = None

    def type(self):
        return self._type_layer

    def send(self, in_data: Tensor):
        self._cache = in_data

        return Tensor.from_ndarray(cpu.sigmoid2(signals=in_data.to_ndarray()))

    def learn(self, gradients: Tensor):
        return Tensor.from_ndarray(cpu.sigmoid2_derivate(
            signals=self._cache.to_ndarray(),
            alphas=gradients.to_ndarray()
        ))
