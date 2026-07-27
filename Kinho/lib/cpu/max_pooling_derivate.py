import numpy as np

from ...tensor import Tensor
from ._np_bridge import max_pool_np


def max_pooling_derivate(in_data: Tensor, gradients: Tensor) -> Tensor:
    """pooling : derivate for max from a 2x2 square

    Args:
        in_data (Tensor): [K][N][M]
        gradients (Tensor): [K][ceil(N/2)][ceil(M/2)]

    Returns:
        Tensor: [K][N][M]
    """
    eps = 1e-9
    arr = in_data.to_ndarray()
    grad_arr = gradients.to_ndarray()
    K, N, M = arr.shape

    maxvals = max_pool_np(arr)  # (K, ceil(N/2), ceil(M/2))

    # broadcast each pooled cell back over the 2x2 block it came from, then
    # crop to (N, M) - handles odd N/M the same way max_pool_np's padding does
    maxvals_up = np.repeat(np.repeat(maxvals, 2, axis=1), 2, axis=2)[:, :N, :M]
    grad_up = np.repeat(np.repeat(grad_arr, 2, axis=1), 2, axis=2)[:, :N, :M]

    is_max = np.abs(arr - maxvals_up) < eps
    ret = np.where(is_max, grad_up, 0.0)

    return Tensor.from_ndarray(ret)
