from ...tensor import Tensor
from ._np_bridge import to_ndarray, to_tensor, max_pool_np


def max_pooling(in_data: Tensor) -> Tensor:
    """pooling : max from a 2x2 square

    Args:
        in_data (Tensor): [k][N][M]

    Returns:
        Tensor: [k]][ceil(N/2)][ceil(M/2)]
    """
    return to_tensor(max_pool_np(to_ndarray(in_data)))
