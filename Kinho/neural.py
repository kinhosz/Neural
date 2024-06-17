import numpy as np
import math
from .transfer import loadTo
from .brain import Wrapper, Builder
from .lib import *
from numba import cuda, NumbaPerformanceWarning
import warnings

EPS = 1e-8

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    def __init__(self, sizes=None, brain_path=None, eta=0.01, gpu=False, mini_batch_size=1, multilabel=False):
        if not sizes and not brain_path:
            raise TypeError('Should set `sizes` or `brain_path` params')

        self._layer = []
        self.__eta = np.array([eta])
        self.__gpuMode = gpu
        self.__mini_batch = mini_batch_size
        self.__fill = 0
        self.__tmp = {}
        self._multilabel = multilabel
        
        if 2**int(math.log2(self.__mini_batch)) != self.__mini_batch:
            raise TypeError("Mini-batch size is invalid. You can use {}".format(2**int(math.log2(max(1, self.__mini_batch)))))

        self.__setBrain(Builder(brain_path) if brain_path else None, sizes)

    def __setBrain(self, brain, sizes):
        if brain:
            self._buildArchitecture(architecture=brain.architecture(),
                                    weights=brain.weights(),
                                    biases=brain.biases())
        else:
            self._buildArchitecture(architecture=sizes,
                                    weights=self._genRandomWeights(sizes),
                                    biases=self._genRandomBiases(sizes))

    def _genRandomWeights(self, arch):
        return [
            np.random.uniform(-2,2,x*y).reshape((x, y)) 
            for x,y in zip(arch[:-1], arch[1:])
        ]
    
    def _genRandomBiases(self, arch):
        return [
            np.random.uniform(0, -1, x).reshape((1,x)) 
            for x in arch[1:]
        ]

    def _buildArchitecture(self, architecture, weights, biases):
        for i in range(0, len(architecture) - 1):
            self._layer.append(
                Sigmoid2(
                    (self.__mini_batch, 1, architecture[i]),
                    (self.__mini_batch, 1, architecture[i]),
                    gpuMode=self.__gpuMode
                )
            )

            self._layer.append(
                DotMatrix(
                    batchsize=self.__mini_batch,
                    weight=weights[i],
                    biase=biases[i],
                    eta=self.__eta,
                    gpuMode=self.__gpuMode
                )
            )
        
        self._layer.append(
            Sigmoid2(
                (self.__mini_batch, 1, architecture[-1]),
                (self.__mini_batch, 1, architecture[-1]),
                gpuMode=self.__gpuMode
            )
        )
        
        if not self._multilabel:
            self._layer.append(
                Softmax(
                    (self.__mini_batch, 1, architecture[-1]),
                    (self.__mini_batch, 1, architecture[-1]),
                    gpuMode=self.__gpuMode
                )
            )
        
        self._layer.append(
            MSE(
               (self.__mini_batch, 1, architecture[-1]),
               gpuMode=self.__gpuMode 
            )
        )

    def __feedForward(self, x):
        for i in range(len(self._layer) - 1):
            x = self._layer[i].send(signals=x)

        return x

    def __backPropagation(self, signals, targets):
        for i in range(len(self._layer) - 1):
            signals = self._layer[i].send(signals)
        
        self._layer[-1].send(signals, targets)
        
        for i in reversed(range(len(self._layer))):
            targets = self._layer[i].learn(targets)
    
    def __buildMsg(self, x):
        if self.__gpuMode:
            return cuda.to_device(x)
        return x
    
    def __insertIntoBatch(self, x, y):
        if 'img' not in self.__tmp.keys():
            self.__tmp['img'] = []
        
        if 'target' not in self.__tmp.keys():
            self.__tmp['target'] = []
        
        self.__tmp['img'].append(x)
        self.__tmp['target'].append(y)

        self.__fill += 1
        
        if self.__fill < self.__mini_batch:
            return None
        
        img = np.array(self.__tmp['img'])
        target = np.array(self.__tmp['target'])
        
        img = self.__buildMsg(img)
        target = self.__buildMsg(target)
        
        self.__backPropagation(img, target)
        self.__fill = 0
        
        self.__tmp['img'] = []
        self.__tmp['target'] = []

    def send(self, input):
        l = input

        x = self.__buildMsg(np.array([[l]]))
        arr = self.__feedForward(x)

        hst, = loadTo(arr, mode='CPU')
        y = hst[0][0]

        return y

    def learn(self, input, output):
        x, y = input, output

        x = np.array([x])
        y = np.array([y])
        
        self.__insertIntoBatch(x, y)

    def cost(self, input, output):
        x, y = input, output

        np_x = self.__buildMsg(np.array([[x]]))
        np_y = self.__buildMsg(np.array([[y]]))
        
        np_x = self.__feedForward(np_x)
        ret = self._layer[-1].send(np_x, np_y)

        return ret[0]
    
    def export(self, filename, path):
        layers = []

        for layer in self._layer:
            if layer.type() == 'neurons':
                layers.append((layer.weight(), layer.bias()))
        
        wrapper = Wrapper(layers)

        with open(path + filename + '.brain', 'wb') as file:
            file.write(wrapper.data())
