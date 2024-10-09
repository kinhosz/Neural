from ...tensor import Tensor
from .get_local_field import get_local_field

def convolution(in_data: Tensor, weight: Tensor, biase: Tensor) -> Tensor:
    """Evaluates the convolution: (weight * in_data) + biase

    Args:
        in_data (Tensor): [K][N][M]
        weight (Tensor): [L][K][3][3]
        biase (Tensor): [L]

    Returns:
        Tensor: [L][N][M]
    """
    _, N, M = in_data.shape()
    L, = biase.shape()

    ret = Tensor((L, N, M))

    for i in range(N):
        for j in range(M):
            local_field = get_local_field(in_data, i, j)
            res = apply_convolution(local_field, weight, biase)
            for l in range(L):
                ret[l][i][j] = res[l]

    return ret

def apply_convolution(local_field: Tensor, weight: Tensor, biase: Tensor) -> Tensor:
    L, = biase.shape()
    res = Tensor((L,))

    for l in range(L):
        res[l] = dot_product(local_field, weight[l]) + biase[l]

    return res

def dot_product(A: Tensor, B: Tensor) -> float:
    K, N, M = A.shape()
    res = 0.0

    for k in range(K):
        for i in range(N):
            for j in range(M):
                res += A[k][i][j] * B[k][i][j]

    return res
