from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_apply_weight_gradients_cpu():
    biase = Tensor((1, 1, 3))
    gradients = Tensor((1, 1, 3))
    eta = 0.1

    biase[0][0][0] = 10
    biase[0][0][1] = 20
    biase[0][0][2] = 30

    gradients[0][0][0] = 5
    gradients[0][0][1] = 10
    gradients[0][0][2] = -10

    cpu.apply_weight_gradients(biase, gradients, eta)

    assert abs(biase[0][0][0] - 9.5) < EPS
    assert abs(biase[0][0][1] - 19) < EPS
    assert abs(biase[0][0][2] - 31) < EPS
