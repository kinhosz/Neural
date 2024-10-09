from ...tensor import Tensor
from .apply_max_pooling import apply_max_pooling

def max_pooling_derivate(in_data: Tensor, gradients: Tensor) -> Tensor:
    """pooling : derivate for max from a 2x2 square

    Args:
        in_data (Tensor): [K][N][M]
        gradients (Tensor): [K][ceil(N/2)][ceil(M/2)]

    Returns:
        Tensor: [K][N][M]
    """
    eps = 1e-9
    K, I, J = in_data.shape()
    ret = Tensor(in_data.shape())

    for k in range(K):
        for i in range(I):
            for j in range(J):
                max_val = apply_max_pooling(in_data, k, i//2, j//2)
                if abs(max_val - in_data[k][i][j]) < eps:
                    ret[k][i][j] = gradients[k][i//2][j//2]
                else:
                    ret[k][i][j] = 0.0

    return ret
