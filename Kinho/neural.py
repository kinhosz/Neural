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
        self._eta = np.array([eta])
        self._gpuMode = gpu
        self._mini_batch = mini_batch_size
        self._fill = 0
        self._tmp = {}
        self._multilabel = multilabel
        
        if 2**int(math.log2(self._mini_batch)) != self._mini_batch:
            raise TypeError("Mini-batch size is invalid. You can use {}".format(2**int(math.log2(max(1, self._mini_batch)))))

        self._setBrain(Builder(brain_path) if brain_path else None, sizes)

    def _setBrain(self, brain, sizes):
        if brain:
            self._buildArchitecture(architecture=brain.architecture(),
                                    weights=brain.weights(),
                                    biases=brain.biases())
        else:
            self._buildArchitecture(architecture=sizes,
                                    weights=self._genRandomWeights(sizes),
                                    biases=self._genRandomBiases(sizes))

    def _genRandomWeights(self, arch):
        # He initialization
        return [
            np.random.normal(0, np.sqrt(2 / x), size=(x, y)) 
            for x, y in zip(arch[:-1], arch[1:])
        ]

    def _genRandomBiases(self, arch):
        return [
            np.zeros((1, x)) for x in arch[1:]
        ]

    def _buildArchitecture(self, architecture, weights, biases):
        for i in range(0, len(architecture) - 1):
            self._layer.append(
                ReLU(
                    (self._mini_batch, 1, architecture[i]),
                    (self._mini_batch, 1, architecture[i]),
                    gpuMode=self._gpuMode
                )
            )

            self._layer.append(
                DotMatrix(
                    batchsize=self._mini_batch,
                    weight=weights[i],
                    biase=biases[i],
                    eta=self._eta,
                    gpuMode=self._gpuMode
                )
            )

        if self._multilabel or architecture[-1] == 1:
            self._layer.append(
                Sigmoid2(
                    (self._mini_batch, 1, architecture[-1]),
                    (self._mini_batch, 1, architecture[-1]),
                    gpuMode=self._gpuMode
                )
            )
            self._layer.append(
                BCE(
                    (self._mini_batch, 1, architecture[-1]),
                    gpuMode=self._gpuMode 
                )
            )
        else:
            self._layer.append(
                Softmax(
                    (self._mini_batch, 1, architecture[-1]),
                    (self._mini_batch, 1, architecture[-1]),
                    gpuMode=self._gpuMode
                )
            )
            # TODO: Change MSE to Cross-entropy
            self._layer.append(
                MSE(
                    (self._mini_batch, 1, architecture[-1]),
                    gpuMode=self._gpuMode 
                )
            )

    def _feedForward(self, x):
        for i in range(len(self._layer) - 1):
            x = self._layer[i].send(signals=x)

        return x

    def _backPropagation(self, signals, targets):
        for i in range(len(self._layer) - 1):
            signals = self._layer[i].send(signals)
        
        self._layer[-1].send(signals, targets)
        
        for i in reversed(range(len(self._layer))):
            targets = self._layer[i].learn(targets)

        return targets
    
    def _buildMsg(self, x):
        if self._gpuMode:
            return cuda.to_device(x)
        return x
    
    def _insertIntoBatch(self, x, y):
        if 'img' not in self._tmp.keys():
            self._tmp['img'] = []
        
        if 'target' not in self._tmp.keys():
            self._tmp['target'] = []
        
        self._tmp['img'].append(x)
        self._tmp['target'].append(y)

        self._fill += 1
        
        if self._fill < self._mini_batch:
            return None
        
        img = np.array(self._tmp['img'])
        target = np.array(self._tmp['target'])
        
        img = self._buildMsg(img)
        target = self._buildMsg(target)
        
        self._backPropagation(img, target)
        self._fill = 0
        
        self._tmp['img'] = []
        self._tmp['target'] = []

    def send(self, input):
        l = input

        x = self._buildMsg(np.array([[l]]))
        arr = self._feedForward(x)

        hst, = loadTo(arr, mode='CPU')
        y = hst[0][0]

        return y

    def learn(self, input, output):
        x, y = input, output

        x = np.array([x])
        y = np.array([y])
        
        self._insertIntoBatch(x, y)

    def cost(self, input, output):
        x, y = input, output

        np_x = self._buildMsg(np.array([[x]]))
        np_y = self._buildMsg(np.array([[y]]))
        
        np_x = self._feedForward(np_x)
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
