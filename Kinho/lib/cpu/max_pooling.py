from ...tensor import Tensor

def max_pooling(in_data: Tensor) -> Tensor:
    """pooling : max from a 2x2 square

    Args:
        in_data (Tensor): [k][N][M]

    Returns:
        Tensor: [k]][ceil(N/2)][ceil(M/2)]
    """
    K, I, J = in_data.shape()
    ret = Tensor((K, (I + 1)//2, (J + 2)//2))
    
    for k in range(K):
        for i in range(0, I, 2):
            new_i = i//2
            for j in range(0, J, 2):
                new_j = j//2
                ret[k][new_i][new_j] = in_data[k][i][j]
                for di in range(2):
                    for dj in range(2):
                        if i + di < I and j + dj < J:
                            ret[k][new_i][new_j] = max(
                                ret[k][new_i][new_j],
                                in_data[k][i + di][j + dj]
                            )
    
    return ret
