import numpy as np
from Deep.lib import GPU_Function as GF
from numba import cuda, float64
from timeit import default_timer as timer
import math
from colorama import Fore, init

init()
MINIMUMBLOCKSIZE = 28

Compiled = {}

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes, eta=0.01, random_weights=True):
        self.__eta = eta
        self.__num_layers = len(sizes)
        self.__sizes = sizes
        self.__weights = None
        self.__weights_device = []
        self.__biases = [np.random.randn(1,x) for x in sizes[1:]]
        self.__biases_device = []
        self.__nablas_w_device = []
        self.__THREADSPERBLOCK = 4096
        self.__THREADSGRID = 16
        self.__stream = cuda.stream()
        self.__eta_device = cuda.to_device(np.array([self.__eta]), stream=self.__stream)
        self.__derror = cuda.to_device(np.zeros([1, self.__sizes[-1]]), stream=self.__stream)

        if random_weights:
            self.__weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.__weights = [np.zeros(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        
        for weigth in self.__weights:
            self.__weights_device.append(cuda.to_device(weigth, stream=self.__stream))
        
        for bias in self.__biases:
            self.__biases_device.append(cuda.to_device(bias, stream=self.__stream))
        
        for l in range(1, self.__num_layers):
            nabla = np.zeros([sizes[l - 1], sizes[l]])
            self.__nablas_w_device.append(cuda.to_device(nabla, stream=self.__stream))
        
        self.__stream.synchronize()
    
    def __getThreads(self, size):
        return pow(2, math.floor(math.log2(size)))

    def __kernelConfig(self, size):
        threads = self.__THREADSPERBLOCK if size > self.__THREADSPERBLOCK else self.__getThreads(size)
        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(size, threads))
        THREADSPERBLOCK = threads

        return (BLOCKSPERGRID, THREADSPERBLOCK)

    def __kernelConfigGrid(self, size_x, size_y):
        blockspergrid_x = max(ceil(size_x, self.__THREADSGRID), MINIMUMBLOCKSIZE)
        blockspergrid_y = max(ceil(size_y, self.__THREADSGRID), MINIMUMBLOCKSIZE)
        BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
        THREADS = (self.__THREADSGRID, self.__THREADSGRID)
        
        return (BLOCKSPERGRID, THREADS)

    def __sync(self, kernel_name):
        t = timer()
        cuda.synchronize()
        t = timer() - t
        ms = round(1000 * t, 3)

        color = Fore.WHITE

        if ms > 1000:
            color = Fore.RED
        elif ms > 1:
            color = Fore.YELLOW
        else:
            color = Fore.GREEN
        
        print(color + "synchronize ({}): {}ms".format(kernel_name, ms))

    def __loss(self, predicted, target):
        result_host = np.zeros(1)
        result_device = cuda.to_device(result_host, stream=self.__stream)

        kernel_name = 'mse'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.mse[self.__kernelConfig(predicted.shape[1])](predicted, target, result_device)
        self.__sync(kernel_name)

        result_host = result_device.copy_to_host(stream=self.__stream)

        return result_host[0]

    def __d_loss(self, y, o):
        loss_host = np.zeros(o.shape)
        loss_device = cuda.to_device(loss_host, stream=self.__stream)

        kernel_name = 'mse_derivate'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.mse_derivate[self.__kernelConfig(o.shape[1])](y, o, loss_device)
        self.__sync(kernel_name)

        return loss_device

    def __selector(self, z):
        result_host = np.zeros(1)
        result_device = cuda.to_device(result_host, stream=self.__stream)

        kernel_name = 'softmax_p1'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.softmax_p1[self.__kernelConfig(z.shape[1])](z, result_device)
        self.__sync(kernel_name)

        kernel_name = 'softmax_p2'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.softmax_p2[self.__kernelConfig(z.shape[1])](z, result_device)
        self.__sync(kernel_name)

        return z

    def __d_selector(self, arr, z, alpha):
        simple_sum_host = np.zeros(1)
        sum_times_alpha_host = np.zeros(1)

        simple_sum = cuda.to_device(simple_sum_host, stream=self.__stream)
        sum_times_alpha = cuda.to_device(sum_times_alpha_host, stream=self.__stream)

        kernel_name = 'softmax_sum_derivate'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.softmax_sum_derivate[self.__kernelConfig(z.shape[1])](arr, z, alpha, simple_sum, sum_times_alpha)
        self.__sync(kernel_name)

        kernel_name = 'softmax_derivate'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.softmax_derivate[self.__kernelConfig(z.shape[1])](arr, z, alpha, simple_sum, sum_times_alpha)
        self.__sync(kernel_name)

    def __activation(self, z):
        kernel_name = 'sigmoid2'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.sigmoid2[self.__kernelConfig(z.shape[1])](z)
        self.__sync(kernel_name)

        return z

    def __d_activation(self, z, alpha):
        kernel_name = 'sigmoid2_derivate'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.sigmoid2_derivate[self.__kernelConfig(z.shape[1])](z, alpha)
        self.__sync(kernel_name)

        return z

    def __layer(self, x, w, b):
        arr_host = np.zeros(b.shape)
        arr = cuda.to_device(arr_host, stream=self.__stream)

        kernel_name = 'dotMatrix'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.dotMatrix[self.__kernelConfigGrid(max(x.shape[0], w.shape[0]), max(x.shape[1], w.shape[1]))](arr, x, w, b)
        self.__sync(kernel_name)

        return arr

    def __d_layer(self, _x, w, alpha):
        arr_host = np.zeros([1, w.shape[0]])
        arr = cuda.to_device(arr_host, stream=self.__stream)

        kernel_name = 'dotMatrix_derivate'
        if kernel_name not in Compiled.keys():
            print(Fore.BLUE + "Compiling {}...".format(kernel_name))
            Compiled[kernel_name] = True

        GF.dotMatrix_derivate[self.__kernelConfigGrid(max(w.shape[0], alpha.shape[0]), max(w.shape[1], alpha.shape[1]))](arr, w, alpha)
        self.__sync(kernel_name)

        return arr
        
    def __feedForward(self, x):
        print(Fore.WHITE+"---------------------------")
        t = timer()
        for w, b in zip(self.__weights_device, self.__biases_device):
            x = self.__activation(x)
            x = self.__layer(x, w, b)
        
        t = timer() - t
        print(Fore.WHITE+"feedForward: {}ms".format(round(1000 * t, 3)))
        print(Fore.WHITE+"---------------------------")

        return self.__selector(x)

    def __backPropagation(self, x, target):
        t = timer()
        print(Fore.WHITE+"------------------")

        t1 = timer()
        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for l in range(len(self.__biases_device)):
            w = self.__weights_device[l]
            b = self.__biases_device[l]
            x = self.__layer(x, w, b)
            activations.append(x)
            x = self.__activation(x)
            z.append(x)

        y = self.__selector(x)

        derror = self.__d_loss(y, target)

        t2 = timer()
        self.__d_selector(self.__derror, z[self.__num_layers - 1], derror)

        t2 = timer() - t2

        error = self.__derror

        t1 = timer() - t1
        print(Fore.WHITE+"part1 - Backprop: {}ms".format(round(t1 * 1000, 3)))
        print(Fore.WHITE+"------------------")

        t1 = timer()
        for l in range(1, self.__num_layers):
            w = self.__weights_device[-l]
            b = self.__biases_device[-l]

            error = self.__d_activation(activations[-l], error)

            coord_x = self.__nablas_w_device[-l].shape[0]
            coord_y = self.__nablas_w_device[-l].shape[1]

            kernel_name = 'transposeDot'
            if kernel_name not in Compiled.keys():
                print(Fore.BLUE + "Compiling {}...".format(kernel_name))
                Compiled[kernel_name] = True

            GF.transposeDot[self.__kernelConfigGrid(coord_x, coord_y)](self.__nablas_w_device[-l], z[-l-1], error)
            self.__sync(kernel_name)

            nabla_b = error # error for each bias
            error = self.__d_layer(z[-l-1], w, error)
            nabla_w  = self.__nablas_w_device[-l]

            kernel_name = 'updateWeights'
            if kernel_name not in Compiled.keys():
                print(Fore.BLUE + "Compiling {}...".format(kernel_name))
                Compiled[kernel_name] = True

            GF.updateWeights[self.__kernelConfigGrid(coord_x, coord_y)](self.__weights_device[-l], self.__eta_device, nabla_w)
            self.__sync(kernel_name)
            GF.updateWeights[self.__kernelConfigGrid(coord_x, coord_y)](self.__biases_device[-l], self.__eta_device, nabla_b)
            self.__sync(kernel_name)
        
        t1 = timer() - t1
        print(Fore.WHITE+"part2 - Backprop: {}ms".format(round(1000 * t1, 3)))
        
        t = timer() - t
        print(Fore.WHITE+"Backpropagation GPU: {}ms".format(round(1000 * t, 3)))
        print(Fore.WHITE+"------------------")

    def send(self, input):
        x_host = np.array([input])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        x_device = self.__activation(x_device)
        
        y_device = self.__feedForward(x_device)
        y_host = y_device.copy_to_host(stream=self.__stream)

        return y_host[0]

    def learn(self, input, output):
        x_host = np.array([input])
        y_host = np.array([output])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        y_device = cuda.to_device(y_host, stream=self.__stream)

        x_device = self.__activation(x_device)
        self.__backPropagation(x_device, y_device)

    def cost(self, input, output):
        x_host = np.array([input])
        target_host = np.array([output])
        x_device = cuda.to_device(x_host, stream=self.__stream)
        target_device = cuda.to_device(target_host, stream=self.__stream)

        x_device = self.__activation(x_device)
        predict_device = self.__feedForward(x_device)

        return self.__loss(predict_device, target_device)    

def main():
    print(Fore.WHITE+"ola mundo")

if __name__ == "__main__":
    main()