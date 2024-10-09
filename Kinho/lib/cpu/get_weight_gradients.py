from ...tensor import Tensor
from .get_local_field import get_local_field

def get_weight_gradients(in_data: Tensor, gradients: Tensor) -> Tensor:
    """Returns gradients for weights

    Args:
        in_data (Tensor): [K][N][M]
        gradients (Tensor): [L][N][M]

    Returns:
        Tensor: [L][K][3][3]
    """
    K, N, M = in_data.shape()
    L, _, _ = gradients.shape()
    ret = Tensor((L, K, 3, 3))

    for i in range(N):
        for j in range(M):
            local_field = get_local_field(in_data, i, j)
            for l in range(L):
                acum_result(ret[l], (local_field * gradients[l][i][j]))

    return ret

def acum_result(ret: Tensor, A: Tensor):
    K, _, _ = ret.shape()

    for k in range(K):
        for i in range(3):
            for j in range(3):
                ret[k][i][j] += A[k][i][j]
