import numpy as np

from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor
from .leaky_relu import LEAK


def leaky_relu_derivate(in_data: Tensor, gradients: Tensor, leak: float = LEAK) -> Tensor:
    """evaluates the derivate of leaky_relu:

    f'(x) = 1 if x > 0 else leak

    Args:
        in_data (Tensor): [K][N][M], the cached forward input
        gradients (Tensor): [K][N][M]

    Returns:
        Tensor: [K][N][M]
    """
    arr = to_ndarray(in_data)
    grad_arr = to_ndarray(gradients)

    return to_tensor(np.where(arr > 0, grad_arr, leak * grad_arr))
