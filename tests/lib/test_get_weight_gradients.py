from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_get_weight_gradients_cpu():
    B, K, N, M = (2, 2, 1, 2)
    L = 3

    in_data = Tensor((B, K, N, M))
    gradients = Tensor((B, L, N, M))

    for b in range(B):
        cnt = 1
        for k in range(K):
            for i in range(N):
                for j in range(M):
                    in_data[b][k][i][j] = cnt
                    cnt += 1

        cnt = 1
        for l in range(L):
            for i in range(N):
                for j in range(M):
                    gradients[b][l][i][j] = cnt
                    cnt += 1

    ret = cpu.get_weight_gradients(in_data, gradients)

    assert ret.shape() == (L, K, 3, 3)

    # both batch entries are identical, so the batch-mean must equal the
    # single-sample value: [0, 0, 1, 1]
    expected_result = 1.0 * 1.0 + 2.0 * 2.0

    assert abs(ret[0][0][1][1] - expected_result) < EPS

def test_get_weight_gradients_batch_mean_cpu():
    K, N, M = (1, 1, 1)
    L = 1

    in_data = Tensor((2, K, N, M))
    gradients = Tensor((2, L, N, M))

    in_data[0][0][0][0] = 2.0
    in_data[1][0][0][0] = 6.0
    gradients[0][0][0][0] = 1.0
    gradients[1][0][0][0] = 1.0

    ret = cpu.get_weight_gradients(in_data, gradients)

    # with 1x1 spatial input, only the centered tap [0, 0, 1, 1] sees any
    # signal (the rest come from zero-padding); its value is the batch-mean
    # of (2.0 * 1.0) and (6.0 * 1.0)
    assert abs(ret[0][0][1][1] - 4.0) < EPS
