import numpy as np

from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor

LEAK = 0.01


def leaky_relu(in_data: Tensor, leak: float = LEAK) -> Tensor:
    """evaluates: x if x > 0 else leak * x

    Args:
        in_data (Tensor): [K][N][M]

    Returns:
        Tensor: [K][N][M]
    """
    arr = to_ndarray(in_data)
    return to_tensor(np.where(arr > 0, arr, leak * arr))
