from ...tensor import Tensor

def apply_max_pooling(in_data: Tensor, K: int, I: int, J: int) -> float:
    _, N, M = in_data.shape()

    val = float('-inf')

    for i in [2*I, 2*I + 1]:
        for j in [2*J, 2*J + 1]:
            if i >= N or j >= M:
                continue
            val = max(val, in_data[K][i][j])

    return val
