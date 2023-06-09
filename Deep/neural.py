import numpy as np
from .lib.GPU import *
from timeit import default_timer as timer
from colorama import Fore, init
from .kernel import *
from .transfer import loadTo, preregister

init()
MINIMUMBLOCKSIZE = 28

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes, eta=0.01, random_weights=True, gpu=False):
        self.__eta = np.array([eta])
        self.__num_layers = len(sizes)
        self.__architecture = sizes
        self.__weights = None
        self.__biases = [np.random.randn(1,x) for x in sizes[1:]]
        self.__gpuMode = gpu
        self.__logs = {}
        self.__mapper = {}

        if random_weights:
            self.__weights = [np.random.uniform(-2,2,x*y).reshape((x, y)) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.__weights = [np.ones(x,y) for x,y in zip(sizes[:-1], sizes[1:])]

        t = timer()
        self.__register()
        self.__reserveMemo()
        t = timer() - t
    
    def __getReserve(self, kernel, pos):
        if self.__gpuMode == False:
            return None
        
        return self.__buffer[kernel][pos]
    
    def __reserveArr(self, module_name, shape, streams):
        arr = cuda.device_array(shape, dtype=np.float64)

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

        for stream in streams:
            stream.synchronize()

    def __register(self):
        if self.__gpuMode == False:
            return None
        
        streams = []

        for bias in self.__biases:
            stream = cuda.stream()
            self.__mapper[id(bias)] = cuda.to_device(bias, stream=stream)
            streams.append(stream)
        
        for weight in self.__weights:
            stream = cuda.stream()
            self.__mapper[id(weight)] = cuda.to_device(weight, stream=stream)
            streams.append(stream)
        
        for stream in streams:
            stream.synchronize()
        
        for bias in self.__biases:
            preregister(id(bias))
        for weigth in self.__weights:
            preregister(id(weigth))

    def logs(self):
        return self.__logs

    def __logger(self, method, delta, dbg=False):
        delta = round(delta * 1000, 3)

        color = Fore.GREEN
        if delta >= 1.0:
            color = Fore.YELLOW
        if delta >= 5.0:
            color = Fore.RED
        
        if dbg:
            print(color + "{}: {}ms".format(method, delta))
    
    def __translate(self, *args):
        mapped = []
        for arg in args:
            if cuda.is_cuda_array(arg):
                mapped.append(arg)
            elif id(arg) in self.__mapper.keys():
                mapped.append(self.__mapper[id(arg)])
            else:
                arg_dvc = cuda.to_device(arg)
                self.__mapper[id(arg)] = arg_dvc
                mapped.append(arg_dvc)
                print("id = {}".format(id(arg)))

        return mapped

    def __swapper(self, *args, buffer, GPURunner, CPURunner):
        if self.__gpuMode == False:
            return CPURunner(*args)
        else:
            new_args = self.__translate(*args)

            return GPURunner(*new_args, buffer)

    def __loss(self, predicted, target, buffer=None):
        return self.__swapper(predicted, target, buffer=buffer, GPURunner=loss, CPURunner=mse_cpu)

    def __d_loss(self, predicted, target, buffer=None):
        return self.__swapper(predicted, target, buffer=buffer, GPURunner=dloss, CPURunner=mse_derivate_cpu)

    def __selector(self, z, buffer=None):
        return self.__swapper(z, buffer=buffer, GPURunner=selector, CPURunner=softmax_cpu)

    def __d_selector(self, z, alpha, buffer=None):
        return self.__swapper(z, alpha, buffer=buffer, GPURunner=dselector, CPURunner=softmax_derivate_cpu)

    def __activation(self, z, buffer=None):
       return self.__swapper(z, buffer=buffer, GPURunner=activation, CPURunner=sigmoid2_cpu)

    def __d_activation(self, z, alpha, buffer=None):
        return self.__swapper(z, alpha, buffer=buffer, GPURunner=dactivation, CPURunner=sigmoid2_derivate_cpu)
    
    def __layer(self, x, w, b, buffer=None):
        ret = self.__swapper(x, w, b, buffer=buffer, GPURunner=layer, CPURunner=dotMatrix_cpu)
        return ret

    def __d_layer(self, _, w, alpha, buffer=None):
        return self.__swapper(w, alpha, buffer=buffer, GPURunner=dlayer, CPURunner=dotMatrix_derivate_cpu)
    
    def __updateWeight(self, weights, eta, nabla_w, buffer=None):
        return self.__swapper(weights, eta, nabla_w, buffer=buffer, GPURunner=updateWeight, CPURunner=updateWeights_cpu)
    
    def __transpose(self, z, derror, buffer=None):
        return self.__swapper(z, derror, buffer=buffer, GPURunner=transpose, CPURunner=transposeDot_cpu)

    def __feedForward(self, x):
        t = timer()

        activation_pointer = 1
        layer_pointer = 0

        for w, b in zip(self.__weights,self.__biases):
            arr = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))
            x = self.__layer(arr, w, b, buffer=self.__getReserve('activation', layer_pointer))
            activation_pointer += 1
            layer_pointer += 1

        y = self.__selector(x, buffer=self.__getReserve('selector', 0))

        t = timer() - t
        self.__logger("feedForward", t)
        return y

    def __backPropagation(self, x, target):
        t = timer()

        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        activation_pointer = 1
        layer_pointer = 0

        for w, b in zip(self.__weights, self.__biases):
            x = self.__layer(x, w, b, buffer=self.__getReserve('layer', layer_pointer))
            layer_pointer += 1

            activations.append(x)
            x = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))
            activation_pointer += 1

            z.append(x)

        y = self.__selector(x, buffer=self.__getReserve('selector', 0))

        derror = self.__d_loss(y, target, buffer=self.__getReserve('d_loss', 0))
        derror = self.__d_selector(z[self.__num_layers - 1], derror, buffer=self.__getReserve('d_selector', 0))

        d_activation_pointer = 0
        d_layer_pointer = 0
        transpose_pointer = 0

        for l in range(1, self.__num_layers):
            w = self.__weights[-l]
            b = self.__biases[-l]

            derror = self.__d_activation(activations[-l], derror, buffer=self.__getReserve('d_activation', d_activation_pointer))
            d_activation_pointer += 1
            
            nabla_w = self.__transpose(z[-l-1], derror, buffer=self.__getReserve('transpose', transpose_pointer))
            transpose_pointer += 1

            nabla_b = derror # error for each bias
            derror =  self.__d_layer(z[-l-1], w, derror, buffer=self.__getReserve('d_layer', d_layer_pointer))
            d_layer_pointer += 1

            self.__weights[-l] = self.__updateWeight(self.__weights[-l], self.__eta, nabla_w)

            self.__biases[-l] = self.__updateWeight(self.__biases[-l], self.__eta, nabla_b)
        
        t = timer() - t
        self.__logger("backpropagation", t)

    def send(self, l):
        t = timer()

        x =  self.__activation(np.array([l]), buffer=self.__getReserve('activation', 0))
        arr = self.__feedForward(x)
        hst, = loadTo(arr, mode='CPU')
        y = hst[0]

        t = timer() - t
        self.__logger("send", t)
        return y

    def learn(self, x, y):
        t = timer()

        x = self.__activation(np.array([x]), buffer=self.__getReserve('activation', 0))
        y = np.array([y])
        self.__backPropagation(x, y)

        t = timer() - t
        self.__logger("learn", t)

    def cost(self, x, y):
        t = timer()

        np_x = np.array([x])
        np_y = np.array([y])
        np_x = self.__activation(np_x, buffer=self.__getReserve('activation', 0))

        ret = self.__loss(self.__feedForward(np_x),np_y)

        t = timer() - t
        self.__logger("cost", t)
        return ret
