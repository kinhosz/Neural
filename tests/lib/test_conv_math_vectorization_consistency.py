"""A/B safety net for vectorizing the conv/pooling math in lib/cpu.

Each `ref_*` function below is an independent, pure-Python re-derivation of
the CURRENT (pre-vectorization) semantics of the matching `Kinho.lib.cpu.*`
function - written from scratch against plain nested lists, not against the
Tensor/implementation being changed. Run this file before AND after
vectorizing to make sure the rewrite is behavior-preserving, not just
"doesn't crash".
"""
import random

import pytest

from Kinho.tensor import Tensor
from Kinho.lib import cpu

EPS = 1e-6


def rand_list(shape):
    if len(shape) == 1:
        return [random.uniform(-1, 1) for _ in range(shape[0])]
    return [rand_list(shape[1:]) for _ in range(shape[0])]


def to_tensor(data):
    return Tensor().load_from(data)


def tensor_to_nested(t):
    if len(t.shape()) == 1:
        return [t[i] for i in range(t.shape()[0])]
    return [tensor_to_nested(t[i]) for i in range(t.shape()[0])]


def assert_close(a, b, path=""):
    if isinstance(a, list):
        assert len(a) == len(b), f"len mismatch at {path}: {len(a)} vs {len(b)}"
        for idx, (x, y) in enumerate(zip(a, b)):
            assert_close(x, y, f"{path}[{idx}]")
    else:
        assert abs(a - b) < EPS, f"mismatch at {path}: {a} vs {b}"


# ---------------------------------------------------------------------------
# reference implementations
# ---------------------------------------------------------------------------

def ref_local_field(in_data, i, j):
    K, N, M = len(in_data), len(in_data[0]), len(in_data[0][0])
    field = [[[0.0] * 3 for _ in range(3)] for _ in range(K)]
    for k in range(K):
        for di in range(-1, 2):
            for dj in range(-1, 2):
                ii, jj = i + di, j + dj
                if 0 <= ii < N and 0 <= jj < M:
                    field[k][di + 1][dj + 1] = in_data[k][ii][jj]
    return field


def ref_convolution(in_data, weight, biase):
    K, N, M = len(in_data), len(in_data[0]), len(in_data[0][0])
    L = len(biase)
    ret = [[[0.0] * M for _ in range(N)] for _ in range(L)]
    for i in range(N):
        for j in range(M):
            field = ref_local_field(in_data, i, j)
            for l in range(L):
                s = 0.0
                for k in range(K):
                    for p in range(3):
                        for q in range(3):
                            s += field[k][p][q] * weight[l][k][p][q]
                ret[l][i][j] = s + biase[l]
    return ret


def ref_convolution_derivate(weight, gradients):
    L, K = len(weight), len(weight[0])
    N, M = len(gradients[0]), len(gradients[0][0])
    ret = [[[0.0] * M for _ in range(N)] for _ in range(K)]
    for i in range(N):
        for j in range(M):
            for l in range(L):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ii, jj = i + di, j + dj
                        if not (0 <= ii < N and 0 <= jj < M):
                            continue
                        for k in range(K):
                            ret[k][ii][jj] += weight[l][k][di + 1][dj + 1] * gradients[l][i][j]
    return ret


def ref_get_weight_gradients(in_data, gradients):
    K, N, M = len(in_data), len(in_data[0]), len(in_data[0][0])
    L = len(gradients)
    ret = [[[[0.0] * 3 for _ in range(3)] for _ in range(K)] for _ in range(L)]
    for i in range(N):
        for j in range(M):
            field = ref_local_field(in_data, i, j)
            for l in range(L):
                g = gradients[l][i][j]
                for k in range(K):
                    for p in range(3):
                        for q in range(3):
                            ret[l][k][p][q] += field[k][p][q] * g
    return ret


def ref_get_biase_gradients(gradients):
    L, N, M = len(gradients), len(gradients[0]), len(gradients[0][0])
    ret = [0.0] * L
    for l in range(L):
        for i in range(N):
            for j in range(M):
                ret[l] += gradients[l][i][j]
    return ret


def ref_apply_weight_gradients(weight, gradients, eta):
    L, K = len(weight), len(weight[0])
    N, M = len(weight[0][0]), len(weight[0][0][0])
    return [[[[weight[l][k][p][q] - gradients[l][k][p][q] * eta for q in range(M)]
              for p in range(N)] for k in range(K)] for l in range(L)]


def ref_apply_biase_gradients(biase, gradients, eta):
    return [biase[i] - gradients[i] * eta for i in range(len(biase))]


def ref_max_pooling(in_data):
    K, N, M = len(in_data), len(in_data[0]), len(in_data[0][0])
    I2, J2 = (N + 1) // 2, (M + 1) // 2
    ret = [[[0.0] * J2 for _ in range(I2)] for _ in range(K)]
    for k in range(K):
        for i in range(I2):
            for j in range(J2):
                val = float('-inf')
                for di in (0, 1):
                    for dj in (0, 1):
                        ii, jj = 2 * i + di, 2 * j + dj
                        if ii < N and jj < M:
                            val = max(val, in_data[k][ii][jj])
                ret[k][i][j] = val
    return ret


def ref_max_pooling_derivate(in_data, gradients):
    K, N, M = len(in_data), len(in_data[0]), len(in_data[0][0])
    eps = 1e-9
    maxvals = ref_max_pooling(in_data)
    ret = [[[0.0] * M for _ in range(N)] for _ in range(K)]
    for k in range(K):
        for i in range(N):
            for j in range(M):
                mv = maxvals[k][i // 2][j // 2]
                if abs(mv - in_data[k][i][j]) < eps:
                    ret[k][i][j] = gradients[k][i // 2][j // 2]
                else:
                    ret[k][i][j] = 0.0
    return ret


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

SHAPES = [(1, 1, 3, 3), (2, 3, 4, 4), (3, 2, 5, 5), (2, 4, 5, 4), (1, 1, 1, 1), (3, 3, 7, 6)]


@pytest.mark.parametrize("K,L,N,M", SHAPES)
def test_convolution_matches_reference(K, L, N, M):
    random.seed(0)
    in_data = rand_list((K, N, M))
    weight = rand_list((L, K, 3, 3))
    biase = rand_list((L,))

    expected = ref_convolution(in_data, weight, biase)
    got = cpu.convolution(to_tensor(in_data), to_tensor(weight), to_tensor(biase))

    assert_close(tensor_to_nested(got), expected)


@pytest.mark.parametrize("K,L,N,M", SHAPES)
def test_convolution_derivate_matches_reference(K, L, N, M):
    random.seed(1)
    weight = rand_list((L, K, 3, 3))
    gradients = rand_list((L, N, M))

    expected = ref_convolution_derivate(weight, gradients)
    got = cpu.convolution_derivate(to_tensor(weight), to_tensor(gradients))

    assert_close(tensor_to_nested(got), expected)


@pytest.mark.parametrize("K,L,N,M", SHAPES)
def test_get_weight_gradients_matches_reference(K, L, N, M):
    random.seed(2)
    in_data = rand_list((K, N, M))
    gradients = rand_list((L, N, M))

    expected = ref_get_weight_gradients(in_data, gradients)
    got = cpu.get_weight_gradients(to_tensor(in_data), to_tensor(gradients))

    assert_close(tensor_to_nested(got), expected)


@pytest.mark.parametrize("K,L,N,M", SHAPES)
def test_get_biase_gradients_matches_reference(K, L, N, M):
    random.seed(3)
    gradients = rand_list((L, N, M))

    expected = ref_get_biase_gradients(gradients)
    got = cpu.get_biase_gradients(to_tensor(gradients))

    assert_close(tensor_to_nested(got), expected)


@pytest.mark.parametrize("K,L,N,M", SHAPES)
def test_apply_weight_gradients_matches_reference(K, L, N, M):
    random.seed(4)
    weight = rand_list((L, K, 3, 3))
    gradients = rand_list((L, K, 3, 3))
    eta = 0.37

    expected = ref_apply_weight_gradients(weight, gradients, eta)
    w_tensor = to_tensor(weight)
    cpu.apply_weight_gradients(w_tensor, to_tensor(gradients), eta)

    assert_close(tensor_to_nested(w_tensor), expected)


def test_apply_biase_gradients_matches_reference():
    random.seed(5)
    biase = rand_list((7,))
    gradients = rand_list((7,))
    eta = 0.37

    expected = ref_apply_biase_gradients(biase, gradients, eta)
    b_tensor = to_tensor(biase)
    cpu.apply_biase_gradients(b_tensor, to_tensor(gradients), eta)

    assert_close(tensor_to_nested(b_tensor), expected)


@pytest.mark.parametrize("K,N,M", [(1, 4, 4), (2, 5, 5), (3, 4, 5), (2, 7, 3), (1, 1, 1), (3, 8, 8)])
def test_max_pooling_matches_reference(K, N, M):
    random.seed(6)
    in_data = rand_list((K, N, M))

    expected = ref_max_pooling(in_data)
    got = cpu.max_pooling(in_data=to_tensor(in_data))

    assert_close(tensor_to_nested(got), expected)


@pytest.mark.parametrize("K,N,M", [(1, 4, 4), (2, 5, 5), (3, 4, 5), (2, 7, 3), (1, 1, 1), (3, 8, 8)])
def test_max_pooling_derivate_matches_reference(K, N, M):
    random.seed(7)
    in_data = rand_list((K, N, M))
    I2, J2 = (N + 1) // 2, (M + 1) // 2
    gradients = rand_list((K, I2, J2))

    expected = ref_max_pooling_derivate(in_data, gradients)
    got = cpu.max_pooling_derivate(in_data=to_tensor(in_data), gradients=to_tensor(gradients))

    assert_close(tensor_to_nested(got), expected)
