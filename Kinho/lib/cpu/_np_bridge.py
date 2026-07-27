import numpy as np

from ...tensor import Tensor


def max_pool_np(arr: np.ndarray) -> np.ndarray:
    """max_pooling's forward math, on a raw (K, N, M) ndarray.

    Shared by max_pooling and max_pooling_derivate so the derivate doesn't
    need to round-trip through Tensor to recompute the forward pass.
    """
    K, N, M = arr.shape
    I2, J2 = (N + 1) // 2, (M + 1) // 2
    pad_i, pad_j = 2 * I2 - N, 2 * J2 - M

    padded = np.pad(arr, ((0, 0), (0, pad_i), (0, pad_j)), constant_values=-np.inf)
    return padded.reshape(K, I2, 2, J2, 2).max(axis=(2, 4))
