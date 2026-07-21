from ..tensor import Tensor
from . import cpu
from .cpu._np_bridge import to_ndarray, to_tensor

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

        return to_tensor(cpu.sigmoid2(signals=to_ndarray(in_data)))

    def learn(self, gradients: Tensor):
        return to_tensor(cpu.sigmoid2_derivate(
            signals=to_ndarray(self._cache),
            alphas=to_ndarray(gradients)
        ))
