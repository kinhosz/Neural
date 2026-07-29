from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_convolution_derivate_cpu():
    B, L, K, N, M = (2, 2, 1, 3, 3)

    weight = Tensor((L, K, 3, 3))
    gradients = Tensor((B, L, N, M))

    # identity matrix
    for i in range(3):
        weight[0][0][i][i] = 1.0
        weight[1][0][i][i] = 2.0

    OFFSET = 100
    for b in range(B):
        for l in range(L):
            for i in range(N):
                for j in range(M):
                    gradients[b][l][i][j] = (l * N * M) + (i * M) + j + b * OFFSET

    ret = cpu.convolution_derivate(weight, gradients)

    assert ret.shape() == (B, K, N, M)

    for b in range(B):
        offset = b * OFFSET
        expected_result = (1.0 * (0.0 + offset)) + (1.0 * (4.0 + offset)) + (1.0 * (8.0 + offset))
        expected_result += (2.0 * (9.0 + offset)) + (2.0 * (13.0 + offset)) + (2.0 * (17.0 + offset))

        assert abs(ret[b][0][1][1] - expected_result) < EPS
