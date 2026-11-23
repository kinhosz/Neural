from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_apply_biase_gradients_cpu():
    biase = Tensor((3,))
    gradients = Tensor((3,))
    eta = 0.1

    biase[0] = 10
    biase[1] = 20
    biase[2] = 30

    gradients[0] = 5
    gradients[1] = 10
    gradients[2] = -10

    cpu.apply_biase_gradients(biase, gradients, eta)

    assert abs(biase[0] - 9.5) < EPS
    assert abs(biase[1] - 19) < EPS
    assert abs(biase[2] - 31) < EPS
