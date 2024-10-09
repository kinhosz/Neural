from ...tensor import Tensor

def max_pooling_derivate(in_data: Tensor, gradients: Tensor) -> Tensor:
    """pooling : derivate for max from a 2x2 square

    Args:
        in_data (Tensor): [K][N][M]
        gradients (Tensor): [K][ceil(N/2)][ceil(M/2)]

    Returns:
        Tensor: [K][N][M]
    """
    eps = 1e-9
    K, I, J = gradients.shape()
    ret = Tensor(in_data.shape())
    
    for k in range(K):
        for i in range(I):
            for j in range(J):
                max_val = in_data[k][2*i][2*j]
                for di in [2*i, 2*i + 1]:
                    for dj in [2*j, 2*j + 1]:
                        if di >= I or dj >= J:
                            continue
                        max_val = max(max_val, in_data[k][di][dj])
                for di in [2*i, 2*i + 1]:
                    for dj in [2*j, 2*j + 1]:
                        if di >= I or dj >= J:
                            continue
                        if abs(max_val - in_data[k][di][dj]) < eps:
                            ret[k][di][dj] = gradients[k][i][j]
                        else:
                            ret[k][di][dj] = 0.0

    return ret
