from ...tensor import Tensor
from .apply_max_pooling import apply_max_pooling

def max_pooling(in_data: Tensor) -> Tensor:
    """pooling : max from a 2x2 square

    Args:
        in_data (Tensor): [k][N][M]

    Returns:
        Tensor: [k]][ceil(N/2)][ceil(M/2)]
    """
    K, I, J = in_data.shape()
    ret = Tensor((K, (I + 1)//2, (J + 2)//2))

    ret_K, ret_I, ret_J = ret.shape()

    for k in range(ret_K):
        for i in range(ret_I):
            for j in range(ret_J):
                ret[k][i][j] = apply_max_pooling(in_data, k, i, j)

    return ret
