from ...tensor import Tensor

def apply_weight_gradients(weight: Tensor, gradients: Tensor, eta: float):
    """Apply: weight[l][k][i][j] = weight[l][k][i][j] - gradients[l][k][i][j] * eta

    Args:
        weight (Tensor): [L][K][3][3]
        gradients (Tensor): [L][K][3][3]
        eta (float): constant
    """
    L, K, N, M = weight.shape()
    for l in range(L):
        for k in range(K):
            for i in range(N):
                for j in range(M):
                    weight[l][k][i][j] = weight[l][k][i][j] - gradients[l][k][i][j] * eta
