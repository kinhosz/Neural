from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8
LEAK = 0.01

def test_leaky_relu_cpu():
    in_data = Tensor((1, 2, 2))

    in_data[0][0][0] = 5
    in_data[0][0][1] = -5
    in_data[0][1][0] = 0
    in_data[0][1][1] = -2

    res = cpu.leaky_relu(in_data=in_data, leak=LEAK)

    assert res.shape() == (1, 2, 2)
    assert abs(res[0][0][0] - 5) < EPS
    assert abs(res[0][0][1] - (-5 * LEAK)) < EPS
    assert abs(res[0][1][0] - 0) < EPS
    assert abs(res[0][1][1] - (-2 * LEAK)) < EPS
