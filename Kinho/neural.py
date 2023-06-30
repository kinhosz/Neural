import numpy as np
from .lib.GPU import *
from timeit import default_timer as timer
from colorama import Fore, init
from .kernel import *
from .transfer import loadTo
from .brain import Wrapper, Builder

init()
MINIMUMBLOCKSIZE = 28

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
        self.__mapper = {}
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
        
        for i in range(1, len(self.__architecture) - 1):
            shape = (self.__mini_batch, 1, self.__architecture[i])
            self.__residual.append(np.empty(shape)) # after layer
            self.__residual.append(np.empty(shape)) # after activation
        
        shape = (self.__mini_batch, 1, self.__architecture[-1])
        self.__residual.append(np.empty(shape)) # selector
        
        self.__target = np.empty((self.__mini_batch, self.__architecture[-1]))
        
        if not self.__gpuMode:
            return None
        
        residual_stream = cuda.stream()
        target_stream = cuda.stream()
        
        raw_residual = cuda.to_device(self.__residual, stream=residual_stream)
        raw_target = cuda.to_device(self.__target, stream=target_stream)
        
        residual_stream.synchronize()
        target_stream.synchronize()
        
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
            self.__reserveArr('weight', self.__weights[-i-1].shape, streams)
    
    def __reserveBias(self, streams):
        for i in range(len(self.__biases)):
            self.__reserveArr('bias', self.__biases[-i-1].shape, streams)

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

    def __swapper(self, *args, buffer, GPURunner, CPURunner):
        arr = None
        t = timer()

        name = GPURunner.__name__
        dbg = True if GPURunner.__name__ == 'transpose' else False

        if self.__gpuMode == False:
            arr = CPURunner(*args, minibatch=self.__mini_batch)
        else:
            mapped_args = []
            for arg in args:
                if cuda.is_cuda_array(arg):
                    mapped_args.append(arg)
                else:
                    arg_dvc = cuda.to_device(arg)
                    mapped_args.append(arg_dvc)
                    self.__mapper[id(arg)] = arg_dvc
                    print("register")
                    assert False

            arr = GPURunner(*mapped_args, buffer, minibatch=self.__mini_batch)

        t = timer() - t
        self.__logger(name, t, dbg=False)

        return arr

    def __loss(self, predicted, target, buffer=None):
        return self.__swapper(predicted, target, buffer=buffer, GPURunner=loss, CPURunner=mse_cpu)

    def __d_loss(self, predicted, target, buffer=None):
        return self.__swapper(predicted, target, buffer=buffer, GPURunner=dloss, CPURunner=mse_derivate_cpu)

    def __selector(self, z, buffer=None):
        return self.__swapper(z, self.__getReserve('extra', 0), buffer=buffer, GPURunner=selector, CPURunner=softmax_cpu)

    def __d_selector(self, z, alpha, buffer=None):
        return self.__swapper(z, alpha, buffer=buffer, GPURunner=dselector, CPURunner=softmax_derivate_cpu)

    def __activation(self, z, buffer=None):
       return self.__swapper(z, buffer=buffer, GPURunner=activation, CPURunner=sigmoid2_cpu)

    def __d_activation(self, z, alpha, buffer=None):
        return self.__swapper(z, alpha, buffer=buffer, GPURunner=dactivation, CPURunner=sigmoid2_derivate_cpu)
    
    def __layer(self, x, w, b, buffer=None):
        return self.__swapper(x, w, b, buffer=buffer, GPURunner=layer, CPURunner=dotMatrix_cpu)

    def __d_layer(self, _, w, alpha, buffer=None):
        return self.__swapper(w, alpha, buffer=buffer, GPURunner=dlayer, CPURunner=dotMatrix_derivate_cpu)
    
    def __updateWeight(self, weights, eta, nabla_w, buffer=None):
        return self.__swapper(weights, eta, nabla_w, buffer=buffer, GPURunner=updateWeight, CPURunner=updateWeights_cpu)
    
    def __transpose(self, z, derror, buffer=None):
        return self.__swapper(z, derror, buffer=buffer, GPURunner=transpose, CPURunner=transposeDot_cpu)
    
    def __copyArr(self, arr, input):
        return self.__swapper(arr, input, buffer=arr, GPURunner=copy, CPURunner=copy_cpu)
    
    def __stochastic_gradient_descent(self, gradients, buffer=None):
        return self.__swapper(gradients, buffer=buffer, GPURunner=sgd, CPURunner=sgd_cpu)

    def __feedForward(self, x):
        t = timer()

        activation_pointer = 0
        layer_pointer = 0

        for w, b in zip(self.__weights,self.__biases):
            arr = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))

            x = self.__layer(arr, w, b, buffer=self.__getReserve('layer', layer_pointer))

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
            # implement batch
            x = self.__layer(x, w, b, buffer=self.__getReserve('layer', layer_pointer))
            layer_pointer += 1

            # implement batch
            self.__copyArr(self.__residual[residual_pointer], x)
            residual_pointer += 1
            
            # implement batch
            x = self.__activation(x, buffer=self.__getReserve('activation', activation_pointer))
            activation_pointer += 1

            # implement batch
            self.__copyArr(self.__residual[residual_pointer], x)
            residual_pointer += 1

        # implement batch
        y = self.__selector(x, buffer=self.__getReserve('selector', 0))
        # implement batch
        self.__copyArr(self.__residual[residual_pointer], y)
        residual_pointer += 1

        # backpropagation
        residual_pointer -= 1
        # implement batch
        derror = self.__d_loss(self.__residual[residual_pointer], self.__target, buffer=self.__getReserve('d_loss', 0))
        
        residual_pointer -= 1
        # implement batch
        derror = self.__d_selector(self.__residual[residual_pointer], derror, buffer=self.__getReserve('d_selector', 0))

        d_activation_pointer = 0
        d_layer_pointer = 0
        transpose_pointer = 0
        update_pointer = 0

        for l in range(1, self.__num_layers):
            w = self.__weights[-l]
            b = self.__biases[-l]

            residual_pointer -= 1
            # implement batch
            derror = self.__d_activation(self.__residual[residual_pointer], derror, buffer=self.__getReserve('d_activation', d_activation_pointer))

            d_activation_pointer += 1
            
            residual_pointer -= 1
            # implement batch
            nabla_w = self.__transpose(self.__residual[residual_pointer], derror, buffer=self.__getReserve('transpose', transpose_pointer))
            transpose_pointer += 1

            nabla_b = derror # error for each bias

            # using the same residual pointer of transpose!
            # implement batch
            derror =  self.__d_layer(self.__residual[residual_pointer], w, derror, buffer=self.__getReserve('d_layer', d_layer_pointer))

            d_layer_pointer += 1

            # implement batch
            self.__weights[-l] = self.__updateWeight(
                self.__weights[-l], 
                self.__eta, 
                # implement batch
                self.__stochastic_gradient_descent(nabla_w, buffer=self.__getReserve('weight', update_pointer)))

            # implement batch
            self.__biases[-l] = self.__updateWeight(
                self.__biases[-l], 
                self.__eta, 
                # implement batch
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
        if 'input' not in self.__tmp.keys():
            self.__tmp['input'] = []
        
        if 'target' not in self.__tmp.keys():
            self.__tmp['target'] = []
        
        self.__tmp['input'].append(x)
        self.__tmp['target'].append(y)

        self.__fill += 1
        
        if self.__fill < self.__mini_batch:
            return None
        
        input = np.array(self.__tmp['input'])
        target = np.array(self.__tmp['target'])
        
        input = self.__buildMsg(input)
        target = self.__buildMsg(target)
        
        input = self.__activation(input, buffer=self.__getReserve('activation', 0))
        
        self.__copyArr(self.__residual[0], input)
        self.__copyArr(self.__target, target)
        
        self.__backPropagation()
        self.__fill = 0
        
        self.__tmp['input'] = []
        self.__tmp['target'] = []

    def send(self, l):
        t = timer()

        x =  self.__buildMsg(np.array([l]))
        arr = self.__feedForward(x)

        hst, = loadTo(arr, mode='CPU')
        y = hst[0]

        t = timer() - t
        self.__logger("send", t)

        return y

    def learn(self, x, y):
        t = timer()

        x = np.array([x])
        y = np.array([y])
        
        self.__insertIntoBatch(x, y)

        t = timer() - t
        self.__logger("learn", t)

    @DeprecationWarning
    def cost(self, x, y):
        t = timer()

        np_x = self.__buildMsg(np.array([x]))
        np_y = self.__buildMsg(np.array([y]))
        np_x = self.__activation(np_x, buffer=self.__getReserve('activation', 0))

        ret = self.__loss(self.__feedForward(np_x),np_y)

        t = timer() - t
        self.__logger("cost", t)
        return ret
    
    def export(self, filename, path):
        layers = []

        for w, b in zip(self.__weights, self.__biases):
            w_host, b_host = loadTo(w, b, mode='CPU')
            layers.append((w_host, b_host))
        
        wrapper = Wrapper(layers)

        with open(path + filename + '.brain', 'wb') as file:
            file.write(wrapper.data())
