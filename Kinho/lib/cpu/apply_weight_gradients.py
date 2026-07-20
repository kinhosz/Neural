from ...tensor import Tensor
from ._np_bridge import to_ndarray


def apply_weight_gradients(weight: Tensor, gradients: Tensor, eta: float):
    """Apply: weight[l][k][i][j] = weight[l][k][i][j] - gradients[l][k][i][j] * eta

    Args:
        weight (Tensor): [L][K][3][3]
        gradients (Tensor): [L][K][3][3]
        eta (float): constant
    """
    new_weight = to_ndarray(weight) - to_ndarray(gradients) * eta

    L, K, N, M = new_weight.shape
    for l in range(L):
        for k in range(K):
            for i in range(N):
                for j in range(M):
                    weight[l][k][i][j] = float(new_weight[l][k][i][j])
