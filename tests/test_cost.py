from Kinho import Neural
from .shared import Shared

shared = Shared()

def cost(robot: Neural):
    images = shared.images()
    labels = shared.labels()
    
    error = robot.cost(images[0], shared.densityArr(labels[0], 10))
    
    assert isinstance(error, float)

def test_cost_cpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=False)
    
    cost(robot)

def test_cost_gpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=True)

    cost(robot)
