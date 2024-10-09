from ...tensor import Tensor

def get_local_field(in_data: Tensor, i: int, j: int) -> Tensor:
    """Given an Tensor and return the local field 3x3 with
    the center on (i, j)

    Args:
        in_data (Tensor): [K][N][M]
        i (int): index < N
        j (int): index < M

    Returns:
        Tensor: [K][3][3]
    """
    K, N, M = in_data.shape()
    field = Tensor((K, 3, 3))

    for k in range(K):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                if i + di < 0 or i + di >= N or j + dj < 0 or j + dj >= M:
                    field[k][di + 1][dj + 1] = 0
                else:
                    field[k][di + 1][dj + 1] = in_data[k][i + di][j + dj]

    return field
