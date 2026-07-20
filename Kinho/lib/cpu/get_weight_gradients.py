import numpy as np

from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor


def get_weight_gradients(in_data: Tensor, gradients: Tensor) -> Tensor:
    """Returns gradients for weights

    Args:
        in_data (Tensor): [K][N][M]
        gradients (Tensor): [L][N][M]

    Returns:
        Tensor: [L][K][3][3]
    """
    in_arr = to_ndarray(in_data)
    grad_arr = to_ndarray(gradients)

    K, N, M = in_arr.shape
    L = grad_arr.shape[0]

    # same im2col columns the forward convolution builds from `in_data`;
    # the weight gradient is just the gradient map correlated against them
    padded = np.pad(in_arr, ((0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(N * M, K * 3 * 3)

    grad_flat = grad_arr.reshape(L, N * M)
    weight_grad = (grad_flat @ cols).reshape(L, K, 3, 3)

    return to_tensor(weight_grad)
