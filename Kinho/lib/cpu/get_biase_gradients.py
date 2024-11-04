from ...tensor import Tensor

def get_biase_gradients(gradients: Tensor) -> Tensor:
    """Reduce to an 1D Tensor

    Args:
        gradients (Tensor): [L][N][M]

    Returns:
        Tensor: [L]
    """
    L, N, M = gradients.shape()
    ret = Tensor((L,))

    for l in range(L):
        for i in range(N):
            for j in range(M):
                ret[l] += gradients[l][i][j]

    return ret
