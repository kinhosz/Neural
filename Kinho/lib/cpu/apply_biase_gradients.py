from ...tensor import Tensor

def apply_biase_gradients(biase: Tensor, gradients: Tensor, eta: float):
    """Apply: biase[i] = biase[i] - gradients[i] * eta

    Args:
        biase (Tensor): [N]
        gradients (Tensor): [N]
        eta (float): constant
    """
    N, = biase.shape()
    for i in range(N):
        biase[i] = biase[i] - gradients[i] * eta
