import numpy as np
from .lib.GPU import *
from timeit import default_timer as timer
from colorama import Fore, init
from .device import *

init()
MINIMUMBLOCKSIZE = 28

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes, eta=0.01, random_weights=True, gpu=False):
        self.__eta = np.array([eta])
        self.__num_layers = len(sizes)
        self.__weights = None
        self.__biases = [np.random.randn(1,x) for x in sizes[1:]]
        self.__gpuMode = gpu
        self.__kernelsCompiled = []
        self.__kernelTimer = {}
        self.__lastSwap = timer()

        if random_weights:
            self.__weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.__weights = [np.ones(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
    
    def __logger(self, method, delta, dbg=False):
        delta = round(delta * 1000, 3)

        color = Fore.GREEN
        if delta >= 1.0:
            color = Fore.YELLOW
        if delta >= 5.0:
            color = Fore.RED
        
        if dbg:
            print(color + "{}: {}ms".format(method, delta))

    def __swapper(self, *args, GPURunner, CPURunner):
        if self.__gpuMode == False:
            return CPURunner(*args)
        
        TIME_IN_S = 60 * 1000
        if abs(self.__lastSwap - timer()) > TIME_IN_S:
            self.__kernelTimer.clear()
            self.__lastSwap = timer()
        
        kernel_name = GPURunner.__name__
        if kernel_name not in self.__kernelsCompiled:
            self.__kernelsCompiled.append(kernel_name)
            return GPURunner(*args)
        
        params = ""
        for arg in args:
            params += '$' + str(arg.shape)
        kernelId = kernel_name + '#' + params

        if kernelId not in self.__kernelTimer.keys():
            self.__kernelTimer[kernelId] = {}

            t = timer()
            arr = GPURunner(*args)
            t = timer() - t

            self.__kernelTimer[kernelId]['GPU'] = t
            return arr
        elif 'CPU' not in self.__kernelTimer[kernelId].keys():
            t = timer()
            arr = CPURunner(*args)
            t = timer() - t

            self.__kernelTimer[kernelId]['CPU'] = t
            return arr
        elif self.__kernelTimer[kernelId]['GPU'] < self.__kernelTimer[kernelId]['CPU']:
            return GPURunner(*args)
        else:
            return CPURunner(*args)

    def __loss(self, predicted, target):
        return self.__swapper(predicted, target, GPURunner=loss, CPURunner=mse_cpu)

    def __d_loss(self, predicted, target):
        return self.__swapper(predicted, target, GPURunner=dloss, CPURunner=mse_derivate_cpu)

    def __selector(self, z):
        return self.__swapper(z, GPURunner=selector, CPURunner=softmax_cpu)

    def __d_selector(self, z, alpha):
        return self.__swapper(z, alpha, GPURunner=dselector, CPURunner=softmax_derivate_cpu)

    def __activation(self, z):
       return self.__swapper(z, GPURunner=activation, CPURunner=sigmoid2_cpu)

    def __d_activation(self, z, alpha):
        return self.__swapper(z, alpha, GPURunner=dactivation, CPURunner=sigmoid2_derivate_cpu)
    
    def __layer(self, x, w, b):
        return self.__swapper(x, w, b, GPURunner=layer, CPURunner=dotMatrix_cpu)

    def __d_layer(self, _, w, alpha):
        return self.__swapper(w, alpha, GPURunner=dlayer, CPURunner=dotMatrix_derivate_cpu)
    
    def __updateWeight(self, weights, eta, nabla_w):
        return self.__swapper(weights, eta, nabla_w, GPURunner=updateWeight, CPURunner=updateWeights_cpu)
    
    def __transpose(self, z, derror):
        return self.__swapper(z, derror, GPURunner=transpose, CPURunner=transposeDot_cpu)

    def __feedForward(self, x):
        t = timer()

        for w, b in zip(self.__weights,self.__biases):
            x = self.__layer(self.__activation(x), w, b)

        y = self.__selector(x)

        t = timer() - t
        self.__logger("feedForward", t)
        return y

    def __backPropagation(self, x, target):
        t = timer()

        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for w, b in zip(self.__weights,self.__biases):
            x = self.__layer(x,w,b)
            activations.append(x)
            x = self.__activation(x)
            z.append(x)

        y = self.__selector(x)

        derror = self.__d_loss(y,target)
        derror = self.__d_selector(z[self.__num_layers - 1],derror)

        for l in range(1, self.__num_layers):
            w = self.__weights[-l]
            b = self.__biases[-l]

            derror = self.__d_activation(activations[-l],derror)
            
            nabla_w = self.__transpose(z[-l-1], derror)

            nabla_b = derror # error for each bias
            derror =  self.__d_layer(z[-l-1], w, derror)

            self.__weights[-l] = self.__updateWeight(self.__weights[-l], self.__eta, nabla_w)

            self.__biases[-l] = self.__updateWeight(self.__biases[-l], self.__eta, nabla_b)
        
        t = timer() - t
        self.__logger("backpropagation", t)

    def send(self, l):
        t = timer()

        x =  self.__activation(np.array([l]))
        y = self.__feedForward(x)[0]

        t = timer() - t
        self.__logger("send", t)
        return y

    def learn(self, x, y):
        t = timer()

        x = self.__activation(np.array([x]))
        y = np.array([y])
        self.__backPropagation(x,y)

        t = timer() - t
        self.__logger("learn", t)

    def cost(self, x, y):
        t = timer()

        np_x = np.array([x])
        np_y = np.array([y])
        np_x = self.__activation(np_x)

        ret = self.__loss(self.__feedForward(np_x),np_y)

        t = timer() - t
        self.__logger("cost", t)
        return ret
