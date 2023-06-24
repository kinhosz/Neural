from Kinho import Neural
from .shared import Shared

import random

shared = Shared()

LEARN_RATE = 0.6

def learn_rate(robot: Neural, epoch: int):
    images = shared.images()
    labels = shared.labels()
    
    zipped_data = [(img, lbl) for img, lbl in zip(images, labels)]
    random.shuffle(zipped_data)
    
    ALPHA = 0.1
    TEST_SIZE = int(len(zipped_data) * ALPHA)
    
    test = zipped_data[:TEST_SIZE]
    train = zipped_data[TEST_SIZE:]
    
    for i in range(epoch):
        for input in train:
            robot.learn(x=input[0], y=shared.densityArr(input[1], 10))
    
    hits = 0
    for input in test:
        out = robot.send(l=input[0])
        if shared.greaterIdx(out) == input[1]:
            hits += 1
    
    return hits/TEST_SIZE

def test_learn_rate_cpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=False)
    
    assert learn_rate(robot=robot, epoch=1) >= LEARN_RATE

def test_learn_rate_gpu():
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=True)
    
    assert learn_rate(robot=robot, epoch=1) >= LEARN_RATE

