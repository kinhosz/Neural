from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor


def get_biase_gradients(gradients: Tensor) -> Tensor:
    """Reduce to an 1D Tensor

    Args:
        gradients (Tensor): [L][N][M]

    Returns:
        Tensor: [L]
    """
    grad_arr = to_ndarray(gradients)
    return to_tensor(grad_arr.sum(axis=(1, 2)))
