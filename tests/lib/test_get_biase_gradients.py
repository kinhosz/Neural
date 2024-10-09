from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_get_biase_gradients_cpu():
    L, N, M = (2, 2, 2)

    gradients = Tensor((L, N, M))

    for l in range(L):
        for i in range(N):
            for j in range(M):
                gradients[l][i][j] = (l * N * M) + (i * M) + j

    ret = cpu.get_biase_gradients(gradients)

    assert ret.shape() == (L,)

    expected_result_0 = 0.0 + 1.0 + 2.0 + 3.0
    expected_result_1 = 4.0 + 5.0 + 6.0 + 7.0

    assert abs(ret[0] - expected_result_0) < EPS
    assert abs(ret[1] - expected_result_1) < EPS
