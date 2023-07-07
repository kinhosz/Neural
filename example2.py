from Kinho import Neural
from tests.shared import Shared
from timeit import default_timer as timer

import random

shared = Shared()

LEARN_RATE = 0.6

def main():
    LOG_SIZE = 256
    robot = Neural(sizes=[28*28, 15, 10], eta=0.1, gpu=True, mini_batch_size=LOG_SIZE)
    epoch = 1
    
    images = shared.images()
    labels = shared.labels()
    
    zipped_data = [(img, lbl) for img, lbl in zip(images, labels)]
    random.shuffle(zipped_data)
    
    ALPHA = 0.1
    TEST_SIZE = int(len(zipped_data) * ALPHA)
    
    test = zipped_data[:TEST_SIZE]
    train = zipped_data[TEST_SIZE:]
    
    data_cnt = 0

    t = timer()
    for i in range(epoch):
        for input in train:
            robot.learn(input[0], shared.densityArr(input[1], 10))
            
            if data_cnt % LOG_SIZE == 0:
                t = timer() - t
                print("COUNT: {}/{} -> {}ms".format(data_cnt//LOG_SIZE, len(train)//LOG_SIZE, round(1000*t, 3)))
                t = timer()
            data_cnt += 1
    
    hits = 0
    for input in test:
        out = robot.send(input[0])
        if shared.greaterIdx(out) == input[1]:
            hits += 1
    
    print("accuracy: {}".format(hits/TEST_SIZE))

if __name__ == "__main__":
    main()