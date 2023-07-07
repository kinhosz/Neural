from Kinho import Neural
from .shared import Shared
import random

shared = Shared()

def mirror(tmp: Neural):
    images = shared.images()
    labels = shared.labels()
    
    zipped_data = [(img, lbl) for img, lbl in zip(images, labels)]
    random.shuffle(zipped_data)
    
    PARTIAL_TRAIN = int(len(zipped_data) * 0.1)
    
    for input in zipped_data[:PARTIAL_TRAIN]:
        tmp.learn(input[0], shared.densityArr(input[1], 10))
    
    tmp.export(filename='tmp_test', path='./tmp/')
    
    robot_cpu = Neural(brain_path='./tmp/tmp_test.brain', eta=0.01, gpu=False)
    robot_gpu = Neural(brain_path='./tmp/tmp_test.brain', eta=0.01, gpu=True)
    
    random.shuffle(zipped_data)
    
    PARTIAL = 100
    EPS = 1e-8
    
    for i in range(PARTIAL):
        cpu_pred = robot_cpu.send(zipped_data[i][0])
        gpu_pred = robot_gpu.send(zipped_data[i][0])
        
        for c_pred, g_pred in zip(cpu_pred, gpu_pred):
            assert abs(c_pred - g_pred) < EPS

def test_mirror_learnWithGPU_gpu():
    tmp = Neural([28*28, 15, 10], eta=0.01, gpu=True)
    mirror(tmp)

def test_mirror_learnWithCPU_gpu():
    tmp = Neural([28*28, 15, 10], eta=0.01, gpu=False)
    mirror(tmp)
