import numpy as np

from Kinho.lib import cpu

EPS = 1e-8
LEAK = 0.01

def test_leaky_relu_dense_derivate_cpu():
    signals = np.array([[[5.0, -5.0, 3.0, -1.0]]])
    alphas = np.array([[[1.0, 1.0, 2.0, 2.0]]])

    res = cpu.leaky_relu_dense_derivate(signals=signals, alphas=alphas, leak=LEAK)

    assert res.shape == (1, 1, 4)
    assert abs(res[0][0][0] - 1) < EPS
    assert abs(res[0][0][1] - (1 * LEAK)) < EPS
    assert abs(res[0][0][2] - 2) < EPS
    assert abs(res[0][0][3] - (2 * LEAK)) < EPS
