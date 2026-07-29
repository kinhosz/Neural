import numpy as np

from ...tensor import Tensor


def convolution(in_data: Tensor, weight: Tensor, biase: Tensor) -> Tensor:
    """Evaluates the convolution: (weight * in_data) + biase

    Args:
        in_data (Tensor): [BATCH][K][N][M]
        weight (Tensor): [L][K][3][3]
        biase (Tensor): [L]

    Returns:
        Tensor: [BATCH][L][N][M]
    """
    in_arr = in_data.to_ndarray()
    weight_arr = weight.to_ndarray()
    biase_arr = biase.to_ndarray()

    B, K, N, M = in_arr.shape
    L = biase_arr.shape[0]

    padded = np.pad(in_arr, ((0, 0), (0, 0), (1, 1), (1, 1)))
    # im2col: every 3x3xK window, one per (batch, output position), as a matrix row
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(B * N * M, K * 3 * 3)

    weight_flat = weight_arr.reshape(L, K * 3 * 3)
    out = cols @ weight_flat.T + biase_arr  # (B*N*M, L)
    out = out.reshape(B, N, M, L).transpose(0, 3, 1, 2)

    return Tensor.from_ndarray(out)
