from Kinho import Neural
from tests.shared import Shared
from timeit import default_timer as timer
from matplotlib import pyplot as plt

import random

shared = Shared()

def testing(robot, test):
    LEN = len(test)
    i = 0
    
    hits = 0
    for input in test:
        print("                    ", end='\r', flush=True)
        print("{}/{}".format(i, LEN), end='\r', flush=True)
        i += 1
        out = robot.send(input[0])
        if shared.greaterIdx(out) == input[1]:
            hits += 1
    
    return hits/len(test)

def training(robot, time_limit):
    images = shared.images()
    labels = shared.labels()
    
    zipped_data = [(img, lbl) for img, lbl in zip(images, labels)]
    random.shuffle(zipped_data)
    
    ALPHA = 0.01
    TEST_SIZE = int(len(zipped_data) * ALPHA)
    
    test = zipped_data[:TEST_SIZE]
    train = zipped_data[TEST_SIZE:]
    
    train_time = 0.0
    
    x = []
    y = []
    
    t = timer()
    acc = 0.0

    acc = testing(robot, test)
                
    x.append(train_time)
    y.append(acc)
    
    print("{}s -> {}".format(round(train_time, 3), round(acc, 3)))
    
    test_timer = 30 # seconds
    
    delta = 0.0

    while train_time < time_limit or acc > 0.93:
        for input in train:
            t = timer()
            robot.learn(input[0], shared.densityArr(input[1], 10))
            t = timer() - t
            
            delta += t                
            if delta > test_timer:
                acc = testing(robot, test)
                
                train_time += delta
                
                x.append(train_time)
                y.append(acc)
                
                delta = 0.0
                print("{}s -> {}".format(round(train_time, 3), round(acc, 3)), flush=True)
            
            if train_time > time_limit:
                break
    
    return x, y

def main():
    sizes = [28*28, 200, 100, 15, 10]
    TIME_LIMIT = 10 * 60
    
    datas = [
        {'gpu': False, 'minibatch': 1, 'info': 'CPU'},
        {'gpu': True, 'minibatch': 1, 'info': 'GPU - MINIBATCH = 1'},
        {'gpu': True, 'minibatch': 2, 'info': 'GPU - MINIBATCH = 2'},
        {'gpu': True, 'minibatch': 16, 'info': 'GPU - MINIBATCH = 16'},
        {'gpu': True, 'minibatch': 128, 'info': 'GPU - MINIBATCH = 128'},
        {'gpu': True, 'minibatch': 1024, 'info': 'GPU - MINIBATCH = 1024'},
    ]
    
    for data in datas:
        robot = Neural(sizes=sizes, eta=0.1, gpu=data['gpu'], mini_batch_size=data['minibatch'])
        print("------------")
        print("label = {}".format(data['info']))
        print("------------")

        x, y = training(robot, TIME_LIMIT)
        plt.plot(x, y, label=data['info'])
    
    plt.xlabel('Seconds')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()