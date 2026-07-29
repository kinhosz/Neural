import numpy as np

from ...tensor import Tensor
from .leaky_relu import LEAK


def leaky_relu_derivate(in_data: Tensor, gradients: Tensor, leak: float = LEAK) -> Tensor:
    """evaluates the derivate of leaky_relu:

    f'(x) = 1 if x > 0 else leak

    Args:
        in_data (Tensor): [BATCH][K][N][M], the cached forward input
        gradients (Tensor): [BATCH][K][N][M]

    Returns:
        Tensor: [BATCH][K][N][M]
    """
    arr = in_data.to_ndarray()
    grad_arr = gradients.to_ndarray()

    return Tensor.from_ndarray(np.where(arr > 0, grad_arr, leak * grad_arr))
