import numpy as np
from Deep.lib import GPU_Function as GF
from numba import cuda

MINIMUMBLOCKSIZE = 28

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
        self.__THREADSPERBLOCK = 16
        self.__stream = cuda.stream()

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

    def __loss(self, predicted, target):
        result_host = np.zeros(1)
        result_device = cuda.to_device(result_host, stream=self.__stream)

        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(predicted.shape[1], self.__THREADSPERBLOCK))

        GF.mse[BLOCKSPERGRID, self.__THREADSPERBLOCK](predicted, target, result_device)
        result_host = result_device.copy_to_host(stream=self.__stream)

        return result_host[0]

    def __d_loss(self, y, o):
        loss_host = np.zeros(o.shape)
        loss_device = cuda.to_device(loss_host, stream=self.__stream)

        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(o.shape[1], self.__THREADSPERBLOCK))

        GF.mse_derivate[BLOCKSPERGRID, self.__THREADSPERBLOCK](y, o, loss_device)

        return loss_device

    def __selector(self, z):
        result_host = np.zeros(1)
        result_device = cuda.to_device(result_host, stream=self.__stream)

        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(z.shape[1], self.__THREADSPERBLOCK))

        GF.softmax_p1[BLOCKSPERGRID, self.__THREADSPERBLOCK](z, result_device)
        GF.softmax_p2[BLOCKSPERGRID, self.__THREADSPERBLOCK](z, result_device)

        return z

    def __d_selector(self, z, alpha):
        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(z.shape[1], self.__THREADSPERBLOCK))

        simple_sum_host = np.zeros(1)
        sum_times_alpha_host = np.zeros(1)

        simple_sum = cuda.to_device(simple_sum_host, stream=self.__stream)
        sum_times_alpha = cuda.to_device(sum_times_alpha_host, stream=self.__stream)

        GF.softmax_sum_derivate[BLOCKSPERGRID, self.__THREADSPERBLOCK](z, alpha, simple_sum, sum_times_alpha)
        GF.softmax_derivate[BLOCKSPERGRID, self.__THREADSPERBLOCK](z, alpha, simple_sum, sum_times_alpha)

        return z

    def __activation(self, z):
        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(z.shape[1], self.__THREADSPERBLOCK))
        GF.sigmoid2[BLOCKSPERGRID, self.__THREADSPERBLOCK](z)

        return z

    def __d_activation(self, z, alpha):
        BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, ceil(z.shape[1], self.__THREADSPERBLOCK))
        GF.sigmoid2_derivate[BLOCKSPERGRID, self.__THREADSPERBLOCK](z, alpha)

        return z

    def __layer(self, x, w, b):
        grid_x_max = max(x.shape[0], w.shape[0])
        grid_y_max = max(x.shape[1], w.shape[1])
        blockspergrid_x = max(ceil(grid_x_max, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
        blockspergrid_y = max(ceil(grid_y_max, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
        BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
        THREADS = (self.__THREADSPERBLOCK, self.__THREADSPERBLOCK)

        arr_host = np.zeros(b.shape)
        arr = cuda.to_device(arr_host, stream=self.__stream)

        GF.dotMatrix[BLOCKSPERGRID, THREADS](arr, x, w, b)
        
        return arr

    def __d_layer(self, _x, w, alpha):
        grid_x_max = max(w.shape[0], alpha.shape[0])
        grid_y_max = max(w.shape[1], alpha.shape[1])
        blockspergrid_x = max(ceil(grid_x_max, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
        blockspergrid_y = max(ceil(grid_y_max, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
        BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
        THREADS = (self.__THREADSPERBLOCK, self.__THREADSPERBLOCK)

        arr_host = np.zeros([1, w.shape[0]])
        arr = cuda.to_device(arr_host, stream=self.__stream)

        GF.dotMatrix_derivate[BLOCKSPERGRID, THREADS](arr, w, alpha)

        return arr
        
    def __feedForward(self, x):
    
        for w, b in zip(self.__weights_device, self.__biases_device):
            x = self.__activation(x)
            x = self.__layer(x, w, b)

        return self.__selector(x)

    def __backPropagation(self, x, target):

        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for w, b in zip(self.__weights_device, self.__biases_device):
            x = self.__layer(x, w, b)
            activations.append(x)
            x = self.__activation(x)
            z.append(x)

        y = self.__selector(x)

        derror = self.__d_loss(y, target)
        derror = self.__d_selector(z[self.__num_layers - 1], derror)

        for l in range(1, self.__num_layers):
            w = self.__weights_device[-l]
            b = self.__biases_device[-l]

            derror = self.__d_activation(activations[-l], derror)

            coord_x = self.__nablas_w_device[-l].shape[0]
            coord_y = self.__nablas_w_device[-l].shape[1]

            blockspergrid_x = max(ceil(coord_x, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
            blockspergrid_y = max(ceil(coord_y, self.__THREADSPERBLOCK), MINIMUMBLOCKSIZE)
            BLOCKSPERGRID = (blockspergrid_x, blockspergrid_y)
            THREADS = (self.__THREADSPERBLOCK, self.__THREADSPERBLOCK)

            GF.transposeDot[BLOCKSPERGRID, THREADS](self.__nablas_w_device[-l], z[-l-1], derror)

            nabla_b = derror # error for each bias
            derror =  self.__d_layer(z[-l-1], w, derror)
            nabla_w  = self.__nablas_w_device[-l]

            eta_device = cuda.to_device(np.array([self.__eta]), stream=self.__stream)
            GF.updateWeights[BLOCKSPERGRID, THREADS](self.__weights_device[-l], eta_device, nabla_w)
            GF.updateWeights[BLOCKSPERGRID, THREADS](self.__biases_device[-l], eta_device, nabla_b)

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
    print("ola mundo")

if __name__ == "__main__":
    main()