import numpy as np

from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor


def convolution_derivate(weight: Tensor, gradients: Tensor) -> Tensor:
    """Evaluates the convolution's derivate: (weight * in_data) + biase

    Args:
        weight (Tensor): [L][K][3][3]
        gradients (Tensor): [L][N][M]

    Returns:
        new gradients: [K][N][M]
    """
    weight_arr = to_ndarray(weight)
    grad_arr = to_ndarray(gradients)

    L, K = weight_arr.shape[0], weight_arr.shape[1]
    N, M = grad_arr.shape[1], grad_arr.shape[2]

    # this is a convolution of `gradients` (as an L-channel "input") against
    # the weight kernel spatially flipped and with its L/K axes swapped
    weight_flipped = weight_arr[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)

    padded = np.pad(grad_arr, ((0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(N * M, L * 3 * 3)

    weight_flat = weight_flipped.reshape(K, L * 3 * 3)
    out = cols @ weight_flat.T  # (N*M, K)
    out = out.T.reshape(K, N, M)

    return to_tensor(out)
