import numpy as np
import math
from timeit import default_timer as timer
from colorama import Fore, init
from .transfer import loadTo
from .brain import Wrapper, Builder
from .lib import cpu, gpu
from numba.cuda.cudadrv.devicearray import DeviceNDArray
from numba import cuda
from typing import Union

DATASTREAM = Union[DeviceNDArray, np.ndarray]

init()
MINIMUMBLOCKSIZE = 28

EPS = 1e-8

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes=None, brain_path=None, eta=0.01, gpu=False, mini_batch_size=1):
        if not sizes and not brain_path:
            raise TypeError('Should set `sizes` or `brain_path` params')
        
        self.__eta = np.array([eta])
        self.__num_layers = None
        self.__architecture = None
        self.__weights = None
        self.__biases = None
        self.__gpuMode = gpu
        self.__mini_batch = mini_batch_size
        self.__residual = None
        self.__target = None
        self.__fill = 0
        self.__logs = {}
        self.__tmp = {}

        t = timer()
        self.__setBrain(Builder(brain_path) if brain_path else None, sizes)
        self.__setMiniBatchMemo()
        self.__reserveMemo()
        t = timer() - t
    
    def __setBrain(self, brain, sizes):
        if brain:
            self.__num_layers = brain.size()
            self.__architecture = brain.architecture()
            self.__biases = brain.biases()
            self.__weights = brain.weights()
        else:
            self.__num_layers = len(sizes)
            self.__architecture = sizes
            self.__biases = [np.random.uniform(0, -1, x).reshape((1,x)) for x in self.__architecture[1:]]
            self.__weights = [np.random.uniform(-2,2,x*y).reshape((x, y)) for x,y in zip(self.__architecture[:-1], self.__architecture[1:])]

        if not self.__gpuMode:
            return None

        self.__eta = cuda.to_device(self.__eta)
        cuda_biases = [cuda.to_device(bias) for bias in self.__biases]
        self.__biases = cuda_biases

        cuda_weights = [cuda.to_device(weight) for weight in self.__weights]
        self.__weights = cuda_weights

    def __setMiniBatchMemo(self):
        if 2**int(math.log2(self.__mini_batch)) != self.__mini_batch:
            raise TypeError("Mini-batch size is invalid. You can use {}".format(2**int(math.log2(self.__mini_batch))))
        
        layers = len(self.__weights)
        activations = layers + 1
        selectors = 1
        
        edges = layers + activations + selectors
        
        self.__residual = []
        
        self.__residual.append(np.empty((self.__mini_batch, 1, self.__architecture[0])))
        
        for i in range(1, len(self.__architecture)):
            shape = (self.__mini_batch, 1, self.__architecture[i])
            self.__residual.append(np.empty(shape)) # after layer
            self.__residual.append(np.empty(shape)) # after activation
        
        shape = (self.__mini_batch, 1, self.__architecture[-1])
        self.__residual.append(np.empty(shape)) # selector
        
        self.__target = np.empty((self.__mini_batch, self.__architecture[-1]))
        
        if not self.__gpuMode:
            return None
        
        raw_residual = []
        
        for res in self.__residual:
            raw_residual.append(
                cuda.to_device(res)
            )
        
        raw_target = cuda.to_device(self.__target)
        
        self.__target = raw_target
        self.__residual = raw_residual
    
    def __getReserve(self, kernel, pos):
        if self.__gpuMode == False:
            return None
        
        return self.__buffer[kernel][pos]
    
    def __addMiniBatchOnShape(self, shape):
        new_shape = (self.__mini_batch, ) + shape
        return new_shape
    
    def __reserveArr(self, module_name, shape, streams, addMiniBatch=True):
        new_shape = shape
        
        if addMiniBatch:
            new_shape = self.__addMiniBatchOnShape(shape)

        arr = cuda.device_array(new_shape, dtype=np.float64)

        stream = cuda.stream()
        self.__buffer[module_name].append(cuda.to_device(arr, stream=stream))
        streams.append(stream)

    def __reserveActivation(self, streams):
        for size in self.__architecture:
            self.__reserveArr('activation', (1, size), streams)
    
    def __reserveLayer(self, streams):
        for size in self.__architecture[1:]:
            self.__reserveArr('layer', (1, size), streams)
    
    def __reserveSelector(self, streams):
        self.__reserveArr('selector', (1, self.__architecture[-1]), streams)
    
    def __reserveDLoss(self, streams):
        self.__reserveArr('d_loss', (1, self.__architecture[-1]), streams)
    
    def __reserveDSelector(self, streams):
        self.__reserveArr('d_selector', (1, self.__architecture[-1]), streams)

    def __reserveDActivation(self, streams):
        LEN = len(self.__architecture)

        for i in range(LEN-1, -1, -1):
            self.__reserveArr('d_activation', (1, self.__architecture[i]), streams)
    
    def __reserveTranspose(self, streams):
        LEN = len(self.__architecture)

        for i in range(LEN-2, -1, -1):
            self.__reserveArr('transpose', (self.__architecture[i], self.__architecture[i+1]), streams)
    
    def __reserveDLayer(self, streams):
        LEN = len(self.__architecture)

        for i in range(LEN-2, -1, -1):
            self.__reserveArr('d_layer', (1, self.__architecture[i]), streams)

    def __reserveExtra(self, streams):
        self.__reserveArr('extra', (1, ), streams)

    def __reserveWeight(self, streams):
        for i in range(len(self.__weights)):
            self.__reserveArr('weight', self.__weights[-i-1].shape, streams, addMiniBatch=False)
    
    def __reserveBias(self, streams):
        for i in range(len(self.__biases)):
            self.__reserveArr('bias', self.__biases[-i-1].shape, streams, addMiniBatch=False)
    
    def __reserveCost(self, streams):
        self.__reserveArr('cost', (1,), streams, addMiniBatch=False)

    def __reserveMemo(self):
        if self.__gpuMode == False:
            return None

        self.__buffer = {
            'activation': [],
            'layer': [],
            'selector': [],
            'd_loss': [],
            'd_selector': [],
            'd_activation': [],
            'transpose': [],
            'd_layer': [],
            'extra': [],
            'weight': [],
            'bias': [],
            'cost': [],
        }

        streams = []
        self.__reserveActivation(streams)
        self.__reserveLayer(streams)
        self.__reserveSelector(streams)
        self.__reserveDLoss(streams)
        self.__reserveDSelector(streams)
        self.__reserveDActivation(streams)
        self.__reserveTranspose(streams)
        self.__reserveDLayer(streams)
        self.__reserveExtra(streams)
        self.__reserveWeight(streams)
        self.__reserveBias(streams)
        self.__reserveCost(streams)

        for stream in streams:
            stream.synchronize()

    def logs(self):
        return self.__logs

    def __logger(self, method, delta, dbg=False):
        delta = round(delta * 1000, 3)

        mode = '[GPU] ' if self.__gpuMode else '[CPU] '
        name = mode + method

        if name not in self.__logs.keys():
            self.__logs[name] = []
        self.__logs[name].append(delta)

        color = Fore.GREEN
        if delta >= 1.0:
            color = Fore.YELLOW
        if delta >= 5.0:
            color = Fore.RED
        
        if dbg:
            print(color + "{}: {}ms".format(method, delta))

    def __loss(self, 
               predict: DATASTREAM, 
               target: DATASTREAM, 
               buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.mse(predict=predict, target=target, buffer=buffer)
        else:
            return cpu.mse(predict=predict, target=target)

    def __d_loss(self, 
                 predicts: DATASTREAM, 
                 targets: DATASTREAM, 
                 buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.mse_derivate(predicts=predicts, targets=targets, buffer=buffer)
        else:
            return cpu.mse_derivate(predicts=predicts, targets=targets)

    def __selector(self, 
                   signals: DATASTREAM, 
                   buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.softmax(signals=signals, extra=self.__getReserve('extra', 0), buffer=buffer)
        else:
            return cpu.softmax(signals=signals)

    def __d_selector(self, 
                     signals: DATASTREAM, 
                     alphas: DATASTREAM, 
                     buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.softmax_derivate(signals=signals, alphas=alphas, extra=self.__getReserve('extra', 0), buffer=buffer)
        else:
            return cpu.softmax_derivate(signals=signals, alphas=alphas)

    def __activation(self, 
                     signals: DATASTREAM, 
                     buffer: DeviceNDArray =None
        ) -> DATASTREAM:

        if self.__gpuMode:
            return gpu.sigmoid2(signals=signals, buffer=buffer)
        else:
            return cpu.sigmoid2(signals=signals)

    def __d_activation(self, 
                       signals: DATASTREAM, 
                       alphas: DATASTREAM, 
                       buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.sigmoid2_derivate(signals=signals, alphas=alphas, buffer=buffer)
        else:
            return cpu.sigmoid2_derivate(signals=signals, alphas=alphas)
    
    def __layer(self, 
                signals: DATASTREAM, 
                weight: DATASTREAM, 
                biase: DATASTREAM, 
                buffer: DeviceNDArray = None
        ) -> DATASTREAM:

        if self.__gpuMode:
            return gpu.dot_matrix(signals=signals, weight=weight, bias=biase, buffer=buffer)
        else:
            return cpu.dot_matrix(signals=signals, weight=weight, bias=biase)

    def __d_layer(self, 
                  const_weight: DATASTREAM, 
                  alphas: DATASTREAM, 
                  buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.dot_matrix_derivate(const_matrix=const_weight, alphas=alphas, buffer=buffer)
        else:
            return cpu.dot_matrix_derivate(const_matrix=const_weight, alphas=alphas)
    
    def __updateWeight(self, 
                       weight: DATASTREAM, 
                       eta: DATASTREAM, 
                       gradient: DATASTREAM, 
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.partial_gradient(weight=weight, eta=eta, gradient=gradient)
        else:
            return cpu.partial_gradient(weight=weight, eta=eta, gradient=gradient)
    
    def __transpose(self, 
                    signals: DATASTREAM, 
                    alphas: DATASTREAM, 
                    buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.transpose(signals=signals, alphas=alphas, buffer=buffer)
        else:
            return cpu.transpose(signals=signals, alphas=alphas)
    
    def __stochastic_gradient_descent(self, 
                                      gradients: DATASTREAM, 
                                      buffer: DeviceNDArray = None
        ) -> DATASTREAM:
        
        if self.__gpuMode:
            return gpu.stochastic_gradient_descent(gradients=gradients, buffer=buffer)
        else:
            return cpu.stochastic_gradient_descent(gradients=gradients)

    def __feedForward(self, x):
        t = timer()
        
        arr = self.__activation(x, buffer=self.__getReserve('activation', 0))

        activation_pointer = 1
        layer_pointer = 0

        for w, b in zip(self.__weights,self.__biases):
            x = self.__layer(arr, w, b, buffer=self.__getReserve('layer', layer_pointer))

            arr = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))

            activation_pointer += 1
            layer_pointer += 1

        y = self.__selector(x, buffer=self.__getReserve('selector', 0))

        t = timer() - t
        self.__logger("feedForward", t)
        return y

    def __backPropagation(self):
        t = timer()

        # feedForward
        x = self.__residual[0]

        activation_pointer = 1
        layer_pointer = 0
        residual_pointer = 1

        for w, b in zip(self.__weights, self.__biases):
            x = self.__layer(x, w, b, buffer=self.__getReserve('layer', layer_pointer))
            layer_pointer += 1

            self.__residual[residual_pointer] = x
            residual_pointer += 1

            x = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))
            activation_pointer += 1

            self.__residual[residual_pointer] = x
            residual_pointer += 1

        y = self.__selector(x, buffer=self.__getReserve('selector', 0))

        self.__residual[residual_pointer] = y
        residual_pointer += 1

        # backpropagation
        residual_pointer -= 1
        derror = self.__d_loss(self.__residual[residual_pointer], self.__target, buffer=self.__getReserve('d_loss', 0))
        
        residual_pointer -= 1
        derror = self.__d_selector(self.__residual[residual_pointer], derror, buffer=self.__getReserve('d_selector', 0))

        d_activation_pointer = 0
        d_layer_pointer = 0
        transpose_pointer = 0
        update_pointer = 0

        for l in range(1, self.__num_layers):
            w = self.__weights[-l]
            b = self.__biases[-l]

            residual_pointer -= 1
            derror = self.__d_activation(self.__residual[residual_pointer], derror, buffer=self.__getReserve('d_activation', d_activation_pointer))

            d_activation_pointer += 1
            
            residual_pointer -= 1
            nabla_w = self.__transpose(self.__residual[residual_pointer], derror, buffer=self.__getReserve('transpose', transpose_pointer))
            transpose_pointer += 1

            nabla_b = derror # error for each bias

            derror =  self.__d_layer(w, derror, buffer=self.__getReserve('d_layer', d_layer_pointer))

            d_layer_pointer += 1

            self.__weights[-l] = self.__updateWeight(
                self.__weights[-l], 
                self.__eta,
                self.__stochastic_gradient_descent(nabla_w, buffer=self.__getReserve('weight', update_pointer)))

            self.__biases[-l] = self.__updateWeight(
                self.__biases[-l], 
                self.__eta, 
                self.__stochastic_gradient_descent(nabla_b, buffer=self.__getReserve('bias', update_pointer)))
            
            update_pointer += 1

        t = timer() - t
        prefix = '[GPU] ' if self.__gpuMode else '[CPU] '
        self.__logger(prefix + "backpropagation", t, dbg=False)
    
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
        
        img = self.__activation(img, buffer=self.__getReserve('activation', 0))
        
        self.__residual[0] = img
        self.__target = target
        
        self.__backPropagation()
        self.__fill = 0
        
        self.__tmp['img'] = []
        self.__tmp['target'] = []

    def send(self, input):
        l = input
        
        t = timer()

        x = self.__buildMsg(np.array([[l]]))
        arr = self.__feedForward(x)

        hst, = loadTo(arr, mode='CPU')
        y = hst[0][0]

        t = timer() - t
        self.__logger("send", t)

        return y

    def learn(self, input, output):
        x, y = input, output

        t = timer()

        x = np.array([x])
        y = np.array([y])
        
        self.__insertIntoBatch(x, y)

        t = timer() - t
        self.__logger("learn", t)

    def cost(self, input, output):
        x, y = input, output
        
        t = timer()

        np_x = self.__buildMsg(np.array([[x]]))
        np_y = self.__buildMsg(np.array([[y]]))

        ret = self.__loss(self.__feedForward(np_x), np_y, buffer=self.__getReserve('cost', 0))

        t = timer() - t
        self.__logger("cost", t)
        return ret[0]
    
    def export(self, filename, path):
        layers = []

        for w, b in zip(self.__weights, self.__biases):
            w_host, b_host = loadTo(w, b, mode='CPU')
            layers.append((w_host, b_host))
        
        wrapper = Wrapper(layers)

        with open(path + filename + '.brain', 'wb') as file:
            file.write(wrapper.data())
