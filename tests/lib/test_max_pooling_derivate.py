from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_max_pooling_derivate_cpu():
    in_data = Tensor((1, 3, 3))
    gradients = Tensor((1, 2, 2))

    in_data[0][0][0] = 10
    in_data[0][0][1] = 10
    in_data[0][0][2] = -10
    in_data[0][1][0] = -3
    in_data[0][1][1] = 5
    in_data[0][1][2] = 4
    in_data[0][2][0] = 8
    in_data[0][2][1] = -1
    in_data[0][2][2] = 7

    gradients[0][0][0] = 10
    gradients[0][0][1] = 5
    gradients[0][1][0] = 7
    gradients[0][1][1] = 15

    res = cpu.max_pooling_derivate(in_data=in_data, gradients=gradients)

    assert res.shape() == in_data.shape()
    assert abs(res[0][0][0] - 10) < EPS
    assert abs(res[0][0][1] - 10) < EPS
    assert abs(res[0][0][2] - 0) < EPS
    assert abs(res[0][1][0] - 0) < EPS
    assert abs(res[0][1][1] - 0) < EPS
    assert abs(res[0][1][2] - 5) < EPS
    assert abs(res[0][2][0] - 7) < EPS
    assert abs(res[0][2][1] - 0) < EPS
    assert abs(res[0][2][2] - 15) < EPS
