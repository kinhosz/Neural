import numpy as np
from .lib.GPU import *
from numba import cuda
from timeit import default_timer as timer
import math
from colorama import Fore, init
from Deep.lib.CPU import Function as CF

init()
MINIMUMBLOCKSIZE = 28

def ceil(A, B):
    return (A + B - 1) // B

class Neural(object):

    def __init__(self, sizes, eta=0.01, random_weights=True):
        self.eta = np.array([eta])
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = None
        self.biases = [np.random.randn(1,x) for x in sizes[1:]]

        if random_weights:
            self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.zeros(x,y) for x,y in zip(sizes[:-1], sizes[1:])]

    ############################################################################
    ############# NETWORK ######################################################

    def __loss(self, predicted, target):
        predicted_dvc = cuda.to_device(predicted)
        target_dvc = cuda.to_device(target)
        arr_dvc = cuda.to_device(np.zeros(1))

        mse[kernelConfig1D(predicted_dvc.shape[0])](arr_dvc, predicted_dvc, target_dvc)
        cuda.synchronize()
        arr = arr_dvc.copy_to_host()
        return arr

    def __d_loss(self, predicted, target):
        LEN = predicted.shape[1]
        predicted_dvc = cuda.to_device(predicted)
        target_dvc = cuda.to_device(target)
        arr_dvc = cuda.to_device(np.zeros([1, LEN]))

        mse_derivate[kernelConfig1D(LEN)](arr_dvc, predicted_dvc, target_dvc)
        cuda.synchronize()
        arr = arr_dvc.copy_to_host()
        return arr

    def __selector(self, z):
        LEN = z.shape[1]
        z_dvc = cuda.to_device(z)
        arr_dvc = cuda.to_device(np.random.randn(1, LEN))
        res_dvc = cuda.to_device(np.zeros(1))

        softmax_p1[kernelConfig1D(LEN)](arr_dvc, z_dvc, res_dvc)
        cuda.synchronize()
        softmax_p2[kernelConfig1D(LEN)](arr_dvc, res_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr

    def __d_selector(self, z, alpha):
        LEN = z.shape[1]

        z_dvc = cuda.to_device(z)
        alpha_dvc = cuda.to_device(alpha)
        arr_dvc = cuda.to_device(np.random.randn(1, LEN))
        ss_dvc = cuda.to_device(np.zeros(1))
        st_dvc = cuda.to_device(np.zeros(1))

        softmax_sum_derivate[kernelConfig1D(LEN)](arr_dvc, z_dvc, alpha_dvc, ss_dvc, st_dvc)
        cuda.synchronize()
        arr = arr_dvc.copy_to_host()
        softmax_derivate[kernelConfig1D(LEN)](arr_dvc, alpha_dvc, ss_dvc, st_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr

    def __activation(self, z):
        LEN = z.shape[1]

        z_dvc = cuda.to_device(z)
        arr_dvc = cuda.to_device(np.random.randn(1, LEN))

        sigmoid2[kernelConfig1D(LEN)](arr_dvc, z_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr

    def __d_activation(self, z, alpha):
        LEN = z.shape[1]

        z_dvc = cuda.to_device(z)
        alpha_dvc = cuda.to_device(alpha)
        arr_dvc = cuda.to_device(np.zeros([1, LEN]))

        sigmoid2_derivate[kernelConfig1D(LEN)](arr_dvc, z_dvc, alpha_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr
    
    def __layer(self, x, w, b):
        LEN = w.shape[1]

        arr_dvc = cuda.to_device(np.zeros([1, LEN]))
        x_dvc = cuda.to_device(x)
        w_dvc = cuda.to_device(w)
        b_dvc = cuda.to_device(b)

        dotMatrix[kernelConfig1D(LEN)](arr_dvc, x_dvc, w_dvc, b_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr

    def __d_layer(self, _x, w, alpha):
        LEN = w.shape[0]
        LEN2 = w.shape[1]

        w_dvc = cuda.to_device(w)
        alpha_dvc = cuda.to_device(alpha)
        arr_dvc = cuda.to_device(np.zeros([1, LEN]))

        dotMatrix_derivate[kernelConfig3D(1, LEN, LEN2)](arr_dvc, w_dvc, alpha_dvc)
        cuda.synchronize()

        arr = arr_dvc.copy_to_host()
        return arr
        
    def __feedForward(self,x):

        for w, b in zip(self.weights,self.biases):
            x = self.__layer(self.__activation(x),w,b)

        return self.__selector(x)

    def __backPropagation(self,x,target):
        t = timer()

        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for w, b in zip(self.weights,self.biases):
            x = self.__layer(x,w,b)
            activations.append(x)
            x = self.__activation(x)
            z.append(x)

        y = self.__selector(x)

        derror = self.__d_loss(y,target)
        derror = self.__d_selector(z[self.num_layers - 1],derror)

        for l in range(1, self.num_layers):
            w = self.weights[-l]
            b = self.biases[-l]

            derror = self.__d_activation(activations[-l],derror)
            
            nabla_w = z[-l-1].transpose().dot(derror) # error for each wij
            LEN1 = z[-l-1].shape[1]
            LEN2 = derror.shape[1]

            arr_dvc = cuda.to_device(np.zeros([LEN1, LEN2]))
            x_dvc = cuda.to_device(z[-l-1])
            derror_dvc = cuda.to_device(derror)

            transposeDot[kernelConfig2D(LEN1, LEN2)](arr_dvc, x_dvc, derror_dvc)
            cuda.synchronize()

            nabla_w = arr_dvc.copy_to_host()

            nabla_b = derror # error for each bias
            derror =  self.__d_layer(z[-l-1], w, derror)

            LEN1, LEN2 = self.weights[-l].shape

            eta_dvc = cuda.to_device(self.eta)
            nabla_w_dvc = cuda.to_device(nabla_w)
            w_dvc = cuda.to_device(self.weights[-l])

            updateWeights[kernelConfig2D(LEN1, LEN2)](w_dvc, eta_dvc, nabla_w_dvc)
            cuda.synchronize()

            self.weights[-l] = w_dvc.copy_to_host()

            LEN1, LEN2 = self.biases[-l].shape

            eta_dvc = cuda.to_device(self.eta)
            nabla_b_dvc = cuda.to_device(nabla_b)
            b_dvc = cuda.to_device(self.biases[-l])

            updateWeights[kernelConfig2D(LEN1, LEN2)](b_dvc, eta_dvc, nabla_b_dvc)
            cuda.synchronize()

            self.biases[-l] = b_dvc.copy_to_host()
        
        t = timer() - t

    def send(self, l):
        x =  self.__activation(np.array([l]))
        return self.__feedForward(x)[0]

    def learn(self,x,y):
        x = self.__activation(np.array([x]))
        y = np.array([y])
        self.__backPropagation(x,y)

    def cost(self,x,y):    
        np_x = np.array([x])
        np_y = np.array([y])
        np_x = self.__activation(np_x)
        return self.__loss(self.__feedForward(np_x),np_y)
