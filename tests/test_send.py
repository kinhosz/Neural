from Kinho import Neural
from .shared import Shared

shared = Shared()

EPS = 1e-8

def send(robot: Neural):
    images = shared.images()
    
    ret = robot.send(input=images[0])
    
    assert len(ret) == 10
    
    acm = 0.0
    for i in range(10):
        acm += ret[i]
    
    assert abs(acm - 1.0) < EPS

def test_send_cpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=False)
    
    send(robot)

def test_send_gpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=True)

    send(robot)
