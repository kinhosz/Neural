import numpy as np

from Kinho.lib import cpu

EPS = 1e-8
LEAK = 0.01

def test_leaky_relu_dense_cpu():
    signals = np.array([[[5.0, -5.0, 0.0, -2.0]]])

    res = cpu.leaky_relu_dense(signals=signals, leak=LEAK)

    assert res.shape == (1, 1, 4)
    assert abs(res[0][0][0] - 5) < EPS
    assert abs(res[0][0][1] - (-5 * LEAK)) < EPS
    assert abs(res[0][0][2] - 0) < EPS
    assert abs(res[0][0][3] - (-2 * LEAK)) < EPS
