from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8
LEAK = 0.01

def test_leaky_relu_derivate_cpu():
    in_data = Tensor((1, 2, 2))
    in_data[0][0][0] = 5
    in_data[0][0][1] = -5
    in_data[0][1][0] = 3
    in_data[0][1][1] = -1

    gradients = Tensor((1, 2, 2))
    gradients[0][0][0] = 1
    gradients[0][0][1] = 1
    gradients[0][1][0] = 2
    gradients[0][1][1] = 2

    res = cpu.leaky_relu_derivate(in_data=in_data, gradients=gradients, leak=LEAK)

    assert res.shape() == (1, 2, 2)
    assert abs(res[0][0][0] - 1) < EPS         # x>0 -> gradient passes through
    assert abs(res[0][0][1] - (1 * LEAK)) < EPS  # x<0 -> gradient scaled by leak
    assert abs(res[0][1][0] - 2) < EPS
    assert abs(res[0][1][1] - (2 * LEAK)) < EPS
