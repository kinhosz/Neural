from Kinho import Tensor

def test_tensor_load_from_cpu():
    l = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

    t = Tensor().load_from(l)

    assert t.shape() == (3, 2, 2)

    for i in range(3):
        for j in range(2):
            for k in range(2):
                assert l[i][j][k] == t[i][j][k]

def test_tensor_mul_cpu():
    t = Tensor((3, 2))

    cnt = 0
    for i in range(3):
        for j in range(2):
            t[i][j] = cnt
            cnt += 1

    t = t * 5

    cnt = 0
    for i in range(3):
        for j in range(2):
            assert t[i][j] == cnt * 5
            cnt += 1
