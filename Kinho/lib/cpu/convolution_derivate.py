import numpy as np

from ...tensor import Tensor


def convolution_derivate(weight: Tensor, gradients: Tensor) -> Tensor:
    """Evaluates the convolution's derivate: (weight * in_data) + biase

    Args:
        weight (Tensor): [L][K][3][3]
        gradients (Tensor): [BATCH][L][N][M]

    Returns:
        new gradients: [BATCH][K][N][M]
    """
    weight_arr = weight.to_ndarray()
    grad_arr = gradients.to_ndarray()

    L, K = weight_arr.shape[0], weight_arr.shape[1]
    B, N, M = grad_arr.shape[0], grad_arr.shape[2], grad_arr.shape[3]

    # this is a convolution of `gradients` (as an L-channel "input") against
    # the weight kernel spatially flipped and with its L/K axes swapped
    weight_flipped = weight_arr[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)

    padded = np.pad(grad_arr, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(padded, (3, 3), axis=(2, 3))
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(B * N * M, L * 3 * 3)

    weight_flat = weight_flipped.reshape(K, L * 3 * 3)
    out = cols @ weight_flat.T  # (B*N*M, K)
    out = out.reshape(B, N, M, K).transpose(0, 3, 1, 2)

    return Tensor.from_ndarray(out)
