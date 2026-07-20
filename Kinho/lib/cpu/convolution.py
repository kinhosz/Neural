import numpy as np

from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor


def convolution(in_data: Tensor, weight: Tensor, biase: Tensor) -> Tensor:
    """Evaluates the convolution: (weight * in_data) + biase

    Args:
        in_data (Tensor): [K][N][M]
        weight (Tensor): [L][K][3][3]
        biase (Tensor): [L]

    Returns:
        Tensor: [L][N][M]
    """
    in_arr = to_ndarray(in_data)
    weight_arr = to_ndarray(weight)
    biase_arr = to_ndarray(biase)

    K, N, M = in_arr.shape
    L = biase_arr.shape[0]

    padded = np.pad(in_arr, ((0, 0), (1, 1), (1, 1)))
    # im2col: every 3x3xK window, one per output position, as a matrix row
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(1, 2))
    cols = cols.transpose(1, 2, 0, 3, 4).reshape(N * M, K * 3 * 3)

    weight_flat = weight_arr.reshape(L, K * 3 * 3)
    out = cols @ weight_flat.T + biase_arr  # (N*M, L)
    out = out.T.reshape(L, N, M)

    return to_tensor(out)
