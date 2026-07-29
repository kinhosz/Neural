from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_get_biase_gradients_cpu():
    B, L, N, M = (2, 2, 2, 2)

    gradients = Tensor((B, L, N, M))

    for b in range(B):
        for l in range(L):
            for i in range(N):
                for j in range(M):
                    gradients[b][l][i][j] = (l * N * M) + (i * M) + j

    ret = cpu.get_biase_gradients(gradients)

    assert ret.shape() == (L,)

    # both batch entries are identical, so the batch-mean must equal the
    # single-sample value
    expected_result_0 = 0.0 + 1.0 + 2.0 + 3.0
    expected_result_1 = 4.0 + 5.0 + 6.0 + 7.0

    assert abs(ret[0] - expected_result_0) < EPS
    assert abs(ret[1] - expected_result_1) < EPS

def test_get_biase_gradients_batch_mean_cpu():
    L, N, M = (1, 1, 1)

    gradients = Tensor((2, L, N, M))
    gradients[0][0][0][0] = 3.0
    gradients[1][0][0][0] = 9.0

    ret = cpu.get_biase_gradients(gradients)

    assert abs(ret[0] - 6.0) < EPS
