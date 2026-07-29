from Kinho import Tensor
from Kinho.lib import cpu

EPS = 1e-8

def test_max_pooling_cpu():
    in_data = Tensor((2, 1, 3, 3))

    in_data[0][0][0][0] = 10
    in_data[0][0][0][1] = 10
    in_data[0][0][0][2] = -10
    in_data[0][0][1][0] = -3
    in_data[0][0][1][1] = 5
    in_data[0][0][1][2] = 4
    in_data[0][0][2][0] = 8
    in_data[0][0][2][1] = -1
    in_data[0][0][2][2] = 7

    # second batch entry: same layout, negated - checks batches don't mix
    in_data[1][0][0][0] = -10
    in_data[1][0][0][1] = -10
    in_data[1][0][0][2] = 10
    in_data[1][0][1][0] = 3
    in_data[1][0][1][1] = -5
    in_data[1][0][1][2] = -4
    in_data[1][0][2][0] = -8
    in_data[1][0][2][1] = 1
    in_data[1][0][2][2] = -7

    res = cpu.max_pooling(in_data=in_data)

    assert res.shape() == (2, 1, 2, 2)

    assert abs(res[0][0][0][0] - 10) < EPS
    assert abs(res[0][0][0][1] - 4) < EPS
    assert abs(res[0][0][1][0] - 8) < EPS
    assert abs(res[0][0][1][1] - 7) < EPS

    assert abs(res[1][0][0][0] - 3) < EPS
    assert abs(res[1][0][0][1] - 10) < EPS
    assert abs(res[1][0][1][0] - 1) < EPS
    assert abs(res[1][0][1][1] - (-7)) < EPS
