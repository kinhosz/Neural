from ...tensor import Tensor
from ._np_bridge import max_pool_np


def max_pooling(in_data: Tensor) -> Tensor:
    """pooling : max from a 2x2 square

    Args:
        in_data (Tensor): [BATCH][K][N][M]

    Returns:
        Tensor: [BATCH][K][ceil(N/2)][ceil(M/2)]
    """
    return Tensor.from_ndarray(max_pool_np(in_data.to_ndarray()))
