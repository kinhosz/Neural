from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_convolution_simple_cpu():
    K, N, M = (1, 3, 3)
    L = 1

    in_data = Tensor((K, N, M))
    weight = Tensor((L, K, 3, 3))
    biase = Tensor((L,))

    weight[0][0][0][0] = 1.0
    weight[0][0][0][1] = 0.0
    weight[0][0][0][2] = 0.0
    weight[0][0][1][0] = 0.0
    weight[0][0][1][1] = 1.0
    weight[0][0][1][2] = 0.0
    weight[0][0][2][0] = 0.0
    weight[0][0][2][1] = 0.0
    weight[0][0][2][2] = 1.0

    in_data[0][0][0] = 1.0
    in_data[0][0][1] = 1.0
    in_data[0][0][2] = 1.0
    in_data[0][1][0] = 2.0
    in_data[0][1][1] = 2.0
    in_data[0][1][2] = 2.0
    in_data[0][2][0] = 3.0
    in_data[0][2][1] = 3.0
    in_data[0][2][2] = 3.0

    biase[0] = 10

    res = cpu.convolution(
        in_data=in_data,
        weight=weight,
        biase=biase
    )

    assert abs(res[0][0][0] - 13) < EPS
    assert abs(res[0][0][1] - 13) < EPS
    assert abs(res[0][0][2] - 11) < EPS
    assert abs(res[0][1][0] - 15) < EPS
    assert abs(res[0][1][1] - 16) < EPS
    assert abs(res[0][1][2] - 13) < EPS
    assert abs(res[0][2][0] - 13) < EPS
    assert abs(res[0][2][1] - 15) < EPS
    assert abs(res[0][2][2] - 15) < EPS

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

    assert abs(val - res[test_L][0][0]) < EPS

    # center test (1, 1) L = 1
    test_L = 1
    val = 0.0

    for k in range(K):
        for i in range(3):
            for j in range(3):
                val += in_data[k][i][j] * weight[test_L][k][i][j]
    val += biase[test_L]

    assert abs(val - res[test_L][1][1]) < EPS
