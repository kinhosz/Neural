import numpy as np

from ...tensor import Tensor


def get_weight_gradients(in_data: Tensor, gradients: Tensor) -> Tensor:
    """Returns gradients for weights: summed over spatial positions (the
    weight is shared across them), averaged over the batch axis.

    Args:
        in_data (Tensor): [BATCH][K][N][M]
        gradients (Tensor): [BATCH][L][N][M]

    Returns:
        Tensor: [L][K][3][3]
    """
    in_arr = in_data.to_ndarray()
    grad_arr = gradients.to_ndarray()

    B, K, N, M = in_arr.shape
    L = grad_arr.shape[1]

    # same im2col columns the forward convolution builds from `in_data`;
    # the weight gradient is just the gradient map correlated against them
    padded = np.pad(in_arr, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(B * N * M, K * 3 * 3)

    grad_flat = grad_arr.transpose(1, 0, 2, 3).reshape(L, B * N * M)
    weight_grad = (grad_flat @ cols).reshape(L, K, 3, 3) / B

    return Tensor.from_ndarray(weight_grad)
