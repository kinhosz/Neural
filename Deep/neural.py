import numpy as np
from .lib.GPU import *
from numba import cuda
from timeit import default_timer as timer
import math
from colorama import Fore, init

init()
MINIMUMBLOCKSIZE = 28

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes, eta=0.01, random_weights=True):
        self.__compiled = {}
        self.__eta = eta
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        self.__weights = None
        self.__weights_device = []
        self.__biases = [np.random.randn(1,x) for x in sizes[1:]]
        self.__biases_device = []
        self.__nablas_w_device = []
        self.__THREADSPERBLOCK = 1024
        self.__stream = cuda.stream()
        self.__eta_device = cuda.to_device(np.array([self.__eta]), stream=self.__stream)
        self.__derror = cuda.to_device(np.zeros([1, self.__sizes[-1]]), stream=self.__stream)
        self.__loss_device = cuda.to_device(np.zeros([1, self.__sizes[-1]]), stream=self.__stream)
        self.__var_layers= []
        self.__var_z = []
        self.__var_activations = []
        self.__var_selector = cuda.to_device(np.zeros([1, self.__sizes[-1]]), stream=self.__stream)
        self.__var_dloss = cuda.to_device(np.zeros([1, self.__sizes[-1]]), stream=self.__stream)

        if random_weights:
            self.__weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.__weights = [np.zeros(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        
        for weigth in self.__weights:
            self.__weights_device.append(cuda.to_device(weigth, stream=self.__stream))
        
        for bias in self.__biases:
            self.__biases_device.append(cuda.to_device(bias, stream=self.__stream))
        
        for l in range(0, self.__num_layers):
            self.__var_layers.append(cuda.to_device(np.zeros([1, self.__sizes[l]]), stream=self.__stream))
            self.__var_z.append(cuda.to_device(np.zeros([1, self.__sizes[l]]), stream=self.__stream))
            self.__var_activations.append(cuda.to_device(np.zeros([1, self.__sizes[l]]), stream=self.__stream))
        
        for l in range(1, self.__num_layers):
            nabla = np.zeros([sizes[l - 1], sizes[l]])
            self.__nablas_w_device.append(cuda.to_device(nabla, stream=self.__stream))
        
        self.__stream.synchronize()

    def __kernelConfig(self, size):
        threads = self.__THREADSPERBLOCK
        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(size, threads))
        THREADSPERBLOCK = threads

        return (BLOCKSPERGRID, THREADSPERBLOCK)
    
    def __divideThreads2(self, size_x, size_y):
        threads = self.__THREADSPERBLOCK
        
        div = [1, 1]
        szs = [size_x, size_y]
        upd = True

        while threads > 1 and upd:
            upd = False
            for i in range(2):
                if div[i] >= szs[i]:
                    continue
                if threads > 1:
                    div[i] *= 2
                    threads /= 2
                    upd = True

        return div[0], div[1]
    
    def __kernelConfigGrid(self, size_x, size_y):
        threads_x, threads_y = self.__divideThreads2(size_x, size_y)

        blockspergrid_x = ceil(size_x, threads_x)
        blockspergrid_y = ceil(size_y, threads_y)

        blocks = blockspergrid_x * blockspergrid_y

        if blocks < MINIMUMBLOCKSIZE:
            add = MINIMUMBLOCKSIZE - blocks
            blocks /= blockspergrid_y
            blockspergrid_y += math.ceil(add / blocks)

        BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
        THREADS = (threads_x, threads_y)
        
        return (BLOCKSPERGRID, THREADS)
    
    def __divideThreads3(self, size_x, size_y, size_z):
        threads = self.__THREADSPERBLOCK
        
        div = [1, 1, 1]
        szs = [size_x, size_y, size_z]
        upd = True

        while threads > 1 and upd:
            upd = False
            for i in range(3):
                if div[i] >= szs[i]:
                    continue
                if threads > 1:
                    div[i] *= 2
                    threads /= 2
                    upd = True

        return div[0], div[1], div[2]
    
    def __kernelConfigGrid3(self, size_x, size_y, size_z):
        threads_x, threads_y, threads_z = self.__divideThreads3(size_x, size_y, size_z)

        blockspergrid_x = ceil(size_x, threads_x)
        blockspergrid_y = ceil(size_y, threads_y)
        blockspergrid_z = ceil(size_z, threads_z)

        blocks = blockspergrid_x * blockspergrid_y * blockspergrid_z

        if blocks < MINIMUMBLOCKSIZE:
            add = MINIMUMBLOCKSIZE - blocks
            blocks /= blockspergrid_z
            blockspergrid_z += math.ceil(add / blocks)

        BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        THREADS = (threads_x, threads_y, threads_z)

        return (BLOCKSPERGRID, THREADS)

    def __sync(self, kernel_name, log=False):
        t = timer()
        cuda.synchronize()
        t = timer() - t
        ms = round(1000 * t, 3)

        color = Fore.WHITE

        if ms > 1000:
            color = Fore.RED
        elif ms > 100:
            color = Fore.YELLOW
        elif ms > 20:
            color = Fore.LIGHTYELLOW_EX
        else:
            color = Fore.GREEN
        
        if log:
            print(color + "synchronize ({}): {}ms".format(kernel_name, ms))
    
    def __isCached(self, kernel_name, log=True):
        if kernel_name not in self.__compiled.keys():
            if log:
                print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            self.__compiled[kernel_name] = True
    
    # write: [arr]
    def __reset(self, arr):
        kernel_name = 'memset'
        if len(arr) == 1:
            self.__isCached(kernel_name)
            memset[self.__kernelConfig(len(arr))](arr)
        else:
            self.__isCached(kernel_name)
            memset2[self.__kernelConfigGrid(arr.shape[0], arr.shape[1])](arr)
        self.__sync(kernel_name)

    # read-only
    def __loss(self, predicted, target):
        result_host = np.zeros(1)
        result_device = cuda.to_device(result_host, stream=self.__stream)

        kernel_name = 'mse'
        self.__isCached(kernel_name)
        mse[self.__kernelConfig(predicted.shape[1])](result_device, predicted, target)
        self.__sync(kernel_name)

        result_host = result_device.copy_to_host(stream=self.__stream)

        return result_host[0]

    # read-only
    def __d_loss(self, y, o):
        loss_device = self.__var_dloss

        kernel_name = 'mse_derivate'
        self.__isCached(kernel_name)
        mse_derivate[self.__kernelConfig(loss_device.shape[1])](loss_device, y, o)
        self.__sync(kernel_name)

        return loss_device

    # write: [arr]
    def __selector(self, arr, x):
        var = cuda.to_device(np.zeros(1))

        kernel_name = 'softmax_p1'
        self.__isCached(kernel_name)
        softmax_p1[self.__kernelConfig(arr.shape[1])](arr, x, var)
        self.__sync(kernel_name)

        kernel_name = 'softmax_p2'
        self.__isCached(kernel_name)
        softmax_p2[self.__kernelConfig(arr.shape[1])](arr, var)
        self.__sync(kernel_name)

    # write: [arr]
    def __d_selector(self, arr, z, alpha):
        simple_sum = cuda.to_device(np.zeros(1))
        sum_times_alpha = cuda.to_device(np.zeros(1))

        kernel_name = 'softmax_sum_derivate'
        self.__isCached(kernel_name)
        softmax_sum_derivate[self.__kernelConfig(z.shape[1])](arr, z, alpha, simple_sum, sum_times_alpha)
        self.__sync(kernel_name)

        kernel_name = 'softmax_derivate'
        self.__isCached(kernel_name)
        softmax_derivate[self.__kernelConfig(z.shape[1])](arr, alpha, simple_sum, sum_times_alpha)
        self.__sync(kernel_name)

    # write: [arr]
    def __activation(self, arr, x):
        kernel_name = 'sigmoid2'
        self.__isCached(kernel_name)
        sigmoid2[self.__kernelConfig(arr.shape[1])](arr, x)
        self.__sync(kernel_name)

    # write: [arr]
    def __d_activation(self, arr, alpha):
        kernel_name = 'sigmoid2_derivate'
        self.__isCached(kernel_name)
        sigmoid2_derivate[self.__kernelConfig(arr.shape[1])](arr, arr, alpha)
        self.__sync(kernel_name)

    # write: [arr]
    def __layer(self, arr, x, w, b):
        self.__reset(arr)

        kernel_name = 'dotMatrix'
        self.__isCached(kernel_name)
        dotMatrix[self.__kernelConfigGrid(arr.shape[0], arr.shape[1])](arr, x, w, b)
        self.__sync(kernel_name)

    # write: [arr]
    def __d_layer(self, arr, w, alpha):
        self.__reset(arr)

        kernel_name = 'dotMatrix_derivate'
        self.__isCached(kernel_name)
        dotMatrix_derivate[self.__kernelConfigGrid3(arr.shape[0], arr.shape[1], alpha.shape[1])](arr, w, alpha)
        self.__sync(kernel_name)
        
    def __feedForward(self, x):
        for l in range(len(self.__biases_device)):
            w, b, arr = self.__weights_device[l], self.__biases_device[l], self.__var_layers[l + 1]
            self.__activation(x, x)
            self.__layer(arr, x, w, b)
            x = arr

        self.__selector(x, x)
        return x

    def __backPropagation(self, x, target):
        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for l in range(len(self.__biases_device)):
            w = self.__weights_device[l]
            b = self.__biases_device[l]
            arr = self.__var_activations[l + 1]
            self.__layer(arr, x, w, b)
            x = arr
            activations.append(x)
            arr = self.__var_z[l + 1]
            self.__activation(arr, x)
            x = arr
            z.append(x)

        arr = self.__var_selector
        self.__selector(arr, x)
        y = arr

        derror = self.__d_loss(y, target)
        error = self.__derror
        self.__d_selector(error, z[self.__num_layers - 1], derror)

        for l in range(1, self.__num_layers):
            w = self.__weights_device[-l]
            b = self.__biases_device[-l]

            self.__d_activation(activations[-l], error)
            error = activations[-l]

            coord_x = self.__nablas_w_device[-l].shape[0]
            coord_y = self.__nablas_w_device[-l].shape[1]

            kernel_name = 'transposeDot'
            self.__isCached(kernel_name)
            transposeDot[self.__kernelConfigGrid(coord_x, coord_y)](self.__nablas_w_device[-l], z[-l-1], error)
            self.__sync(kernel_name)

            nabla_b = error # error for each bias
            arr = self.__var_layers[-l-1]
            self.__d_layer(arr, w, error)
            error = arr
            nabla_w  = self.__nablas_w_device[-l]

            kernel_name = 'updateWeights'
            self.__isCached(kernel_name)
            coord_x, coord_y = self.__weights_device[-l].shape
            updateWeights[self.__kernelConfigGrid(coord_x, coord_y)](self.__weights_device[-l], self.__eta_device, nabla_w)
            self.__sync(kernel_name)
            coord_x, coord_y = self.__biases_device[-l].shape
            updateWeights[self.__kernelConfigGrid(coord_x, coord_y)](self.__biases_device[-l], self.__eta_device, nabla_b)
            self.__sync(kernel_name)

    def send(self, input):
        x_host = np.array([input])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        self.__activation(x_device, x_device)
        
        y_device = self.__feedForward(x_device)
        y_host = y_device.copy_to_host(stream=self.__stream)

        return y_host[0]

    def learn(self, input, output):
        x_host = np.array([input])
        y_host = np.array([output])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        y_device = cuda.to_device(y_host, stream=self.__stream)

        self.__activation(x_device, x_device)
        self.__backPropagation(x_device, y_device)

    def cost(self, input, output):
        x_host = np.array([input])
        target_host = np.array([output])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        target_device = cuda.to_device(target_host, stream=self.__stream)

        self.__activation(x_device, x_device)
        predict_device = self.__feedForward(x_device)

        return self.__loss(predict_device, target_device)    
