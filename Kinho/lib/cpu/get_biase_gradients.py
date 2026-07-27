from ...tensor import Tensor


def get_biase_gradients(gradients: Tensor) -> Tensor:
    """Reduce to an 1D Tensor

    Args:
        gradients (Tensor): [L][N][M]

    Returns:
        Tensor: [L]
    """
    grad_arr = gradients.to_ndarray()
    return Tensor.from_ndarray(grad_arr.sum(axis=(1, 2)))
