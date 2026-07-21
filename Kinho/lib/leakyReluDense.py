from . import cpu

class LeakyReLUDense:
    """LeakyReLU for the dense head, matching Sigmoid2's batched-ndarray interface."""

    def __init__(self, in_shape, out_shape, leak=0.01, gpuMode=False):
        self._gpu = gpuMode
        self._leak = leak
        self._typeLayer = 'activation'
        self._cache = None

    def type(self):
        return self._typeLayer

    def send(self, in_data):
        self._cache = in_data

        return cpu.leaky_relu_dense(signals=in_data, leak=self._leak)

    def learn(self, gradients):
        return cpu.leaky_relu_dense_derivate(
            signals=self._cache,
            alphas=gradients,
            leak=self._leak
        )
