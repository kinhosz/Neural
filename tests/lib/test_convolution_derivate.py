from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_convolution_derivate_cpu():
    L, K, N, M = (2, 1, 3, 3)

    weight = Tensor((L, K, 3, 3))
    gradients = Tensor((L, N, M))

    # identity matrix
    for i in range(3):
        weight[0][0][i][i] = 1.0
        weight[1][0][i][i] = 2.0

    for l in range(L):
        for i in range(N):
            for j in range(M):
                gradients[l][i][j] = (l * N * M) + (i * M) + j
                print(l, i, j, "=", gradients[l][i][j])

    ret = cpu.convolution_derivate(weight, gradients)

    assert ret.shape() == (K, N, M)

    expected_result = (1.0 * 0.0) + (1.0 * 4.0) + (1.0 * 8.0)
    expected_result += (2.0 * 9.0) + (2.0 * 13.0) + (2.0 * 17.0)

    assert abs(ret[0][1][1] - expected_result) < EPS
