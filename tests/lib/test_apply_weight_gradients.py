from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_apply_weight_gradients_cpu():
    weight = Tensor((2, 1, 3, 3))
    gradients = Tensor((2, 1, 3, 3))
    eta = 0.1

    weight[0][0][0][0] = 10
    weight[0][0][0][1] = 20
    weight[1][0][2][2] = 30

    gradients[0][0][0][0] = 5
    gradients[0][0][0][1] = 10
    gradients[1][0][2][2] = -10

    cpu.apply_weight_gradients(weight, gradients, eta)

    assert abs(weight[0][0][0][0] - 9.5) < EPS
    assert abs(weight[0][0][0][1] - 19) < EPS
    assert abs(weight[1][0][2][2] - 31) < EPS
