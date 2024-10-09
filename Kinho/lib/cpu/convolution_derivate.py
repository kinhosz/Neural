from ...tensor import Tensor

def convolution_derivate(weight: Tensor, gradients: Tensor) -> Tensor:
    """Evaluates the convolution's derivate: (weight * in_data) + biase

    Args:
        weight (Tensor): [L][K][3][3]
        gradients (Tensor): [L][N][M]

    Returns:
        new gradients: [K][N][M]
    """
    _, K, _, _ = weight.shape()
    _, N, M = gradients.shape()

    ret = Tensor((K, N, M))

    for i in range(N):
        for j in range(M):
            update_partial_gradient(ret, weight, gradients, i, j)

    return ret

def update_partial_gradient(ret: Tensor, weight: Tensor, gradients: Tensor, i: int, j: int) -> None:
    L, K, _, _ = weight.shape()
    _, N, M = ret.shape()

    for l in range(L):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if di + i < 0 or di + i >= N or dj + j < 0 or dj + j >= M:
                    continue
                for k in range(K):
                    ret[k][di + i][dj + j] += (weight[l][k][di + 1][dj + 1] * gradients[l][i][j])
