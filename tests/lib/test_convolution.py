from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_convolution_cpu():
    K, N, M = (2, 4, 5)
    L = 3

    in_data = Tensor((K, N, M), uniform=True)
    weight = Tensor((L, K, 3, 3), uniform=True)
    biase = Tensor((L,), uniform=True)

    res = cpu.convolution(
        in_data=in_data,
        weight=weight,
        biase=biase
    )

    # border test (0, 0) L = 0
    test_L = 0
    val = 0.0

    for k in range(K):
        for i in range(2):
            for j in range(2):
                val += in_data[k][i][j] * weight[test_L][k][1 + i][1 + j]
    val += biase[test_L]

    assert abs(val - res[test_L]) < EPS

    # center test (1, 1) L = 1
    test_L = 1
    val = 0.0

    for k in range(K):
        for i in range(3):
            for j in range(3):
                val += in_data[k][i][j] * weight[test_L][k][i][j]
    val += biase[test_L]

    assert abs(val - res[test_L]) < EPS
