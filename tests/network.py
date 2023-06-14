from Kinho import CNeural, Neural
from .utils import *
import numpy as np

cpu = None
gpu = None

EPS = 1e-7

def config():
    global cpu
    global gpu

    cpu = CNeural([10, 5, 2], eta=0.1)
    gpu = Neural([10, 5, 2], eta=0.1)

def areEqual(a, b):
    if a.shape != b.shape:
        return False

    L1 = a.shape[0]
    L2 = a.shape[1]

    for x in range(L1):
        for y in range(L2):
            if abs(a[x, y] - b[x, y]) > EPS:
                return False
    
    return True

def loss():
    LEN = 200
    predict = np.random.randn(LEN)
    target = np.random.randn(LEN)

    ans_1 = cpu._Network__loss(predict, target)
    ans_2 = gpu._Neural__loss(predict, target)

    if abs(ans_1 - ans_2) > EPS:
        return False

    return True

def d_loss():
    LEN = 200
    predict = np.random.randn(1, LEN)
    target = np.random.randn(1, LEN)

    ans1 = cpu._Network__d_loss(predict, target)
    ans2 = gpu._Neural__d_loss(predict, target)

    return areEqual(ans1, ans2)

def selector():
    LEN = 200
    z = np.random.randn(1, LEN)

    ans1 = cpu._Network__selector(z)
    ans2 = gpu._Neural__selector(z)

    return areEqual(ans1, ans2)

def d_selector():
    LEN = 200
    z = np.random.randn(1, LEN)
    alpha =  np.random.randn(1, LEN)

    ans1 = cpu._Network__d_selector(z, alpha)
    ans2 = gpu._Neural__d_selector(z, alpha)

    return areEqual(ans1, ans2)

def activation():
    LEN = 200

    z = np.random.randn(1, LEN)
    ans1 = cpu._Network__activation(z)
    ans2 = gpu._Neural__activation(z)

    return areEqual(ans1, ans2)

def d_activation():
    LEN = 200

    z = np.random.randn(1, LEN)
    alpha = np.random.randn(1, LEN)
    
    ans1 = cpu._Network__d_activation(z, alpha)
    ans2 = gpu._Neural__d_activation(z, alpha)

    return areEqual(ans1, ans2)

def layer():
    LEN1 = 200
    LEN2 = 300

    x = np.random.randn(1, LEN1)
    w = np.random.randn(LEN1, LEN2)
    b = np.random.randn(1, LEN2)

    ans1 = cpu._Network__layer(x, w, b)
    ans2 = gpu._Neural__layer(x, w, b)

    return areEqual(ans1, ans2)

def d_layer():
    LEN = 100
    LEN2 = 200

    w = np.random.randn(LEN, LEN2)
    b = np.random.randn(1, LEN2)

    ans1 = cpu._Network__d_layer(w, w, b)
    ans2 = gpu._Neural__d_layer(w, w, b)

    return areEqual(ans1, ans2)

def test():
    config()

    tests = [loss, d_loss, selector, d_selector, activation,
             d_activation, layer, d_layer]
    
    logger("network", tests)