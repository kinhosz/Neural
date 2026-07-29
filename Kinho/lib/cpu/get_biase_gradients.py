from ...tensor import Tensor


def get_biase_gradients(gradients: Tensor) -> Tensor:
    """Reduce to an 1D Tensor: summed over spatial positions (the bias is
    shared across them), averaged over the batch axis.

    Args:
        gradients (Tensor): [BATCH][L][N][M]

    Returns:
        Tensor: [L]
    """
    grad_arr = gradients.to_ndarray()
    B = grad_arr.shape[0]
    return Tensor.from_ndarray(grad_arr.sum(axis=(0, 2, 3)) / B)
