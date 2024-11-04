from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_get_weight_gradients_cpu():
    K, N, M = (2, 1, 2)
    L = 3

    in_data = Tensor((K, N, M))
    gradients = Tensor((L, N, M))

    cnt = 1
    for k in range(K):
        for i in range(N):
            for j in range(M):
                in_data[k][i][j] = cnt
                cnt += 1

    cnt = 1
    for l in range(L):
        for i in range(N):
            for j in range(M):
                gradients[l][i][j] = cnt
                cnt += 1

    ret = cpu.get_weight_gradients(in_data, gradients)

    assert ret.shape() == (L, K, 3, 3)

    # [0, 0, 1, 1]
    expected_result = 1.0 * 1.0 + 2.0 * 2.0

    assert abs(ret[0][0][1][1] - expected_result) < EPS
