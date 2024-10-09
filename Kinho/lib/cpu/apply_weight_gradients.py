from ...tensor import Tensor

def apply_weight_gradients(weight: Tensor, gradients: Tensor, eta: float):
    """Apply: weight[k][i][j] = weight[k][i][j] - gradients[k][i][j] * eta

    Args:
        biase (Tensor): [N]
        gradients (Tensor): [N]
        eta (float): constant
    """
    K, N, M = weight.shape()
    for k in range(K):
        for i in range(N):
            for j in range(M):
                weight[k][i][j] = weight[k][i][j] - gradients[k][i][j] * eta
