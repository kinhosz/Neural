import numpy as np

from ...tensor import Tensor

LEAK = 0.01


def leaky_relu(in_data: Tensor, leak: float = LEAK) -> Tensor:
    """evaluates: x if x > 0 else leak * x

    Args:
        in_data (Tensor): [BATCH][K][N][M]

    Returns:
        Tensor: [BATCH][K][N][M]
    """
    arr = in_data.to_ndarray()
    return Tensor.from_ndarray(np.where(arr > 0, arr, leak * arr))
