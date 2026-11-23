from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_max_pooling_derivate_cpu():
    in_data = Tensor((2, 1, 3, 3))
    gradients = Tensor((2, 1, 2, 2))

    grid = [
        [10, 10, -10],
        [-3, 5, 4],
        [8, -1, 7],
    ]
    grad = [
        [10, 5],
        [7, 15],
    ]

    for b in range(2):
        # same input for both batches, gradients scaled by (b + 1) - checks
        # the mask (derived from in_data) is applied independently per batch
        for i in range(3):
            for j in range(3):
                in_data[b][0][i][j] = grid[i][j]
        for i in range(2):
            for j in range(2):
                gradients[b][0][i][j] = grad[i][j] * (b + 1)

    res = cpu.max_pooling_derivate(in_data=in_data, gradients=gradients)

    assert res.shape() == in_data.shape()

    expected = [
        [10, 10, 0],
        [0, 0, 5],
        [7, 0, 15],
    ]

    for b in range(2):
        for i in range(3):
            for j in range(3):
                assert abs(res[b][0][i][j] - expected[i][j] * (b + 1)) < EPS
