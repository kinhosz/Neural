import numpy as np
from Deep.lib.CPU import Function as CF
from Deep.lib.GPU import Function as GF
from timeit import default_timer as timer

class Network(object):

    def __init__(self,sizes,eta=0.01, random_weights=True, gpu_mode=False):
        self.eta = eta
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = None
        self.biases = [np.random.randn(1,x) for x in sizes[1:]]
        self.gpu_mode = gpu_mode

        if random_weights:
            self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.zeros(x,y) for x,y in zip(sizes[:-1], sizes[1:])]

    ############################################################################
    ############# NETWORK ######################################################

    # funcao para calcular o erro da rede
    # y = minha saida
    # o = output esperado
    def __loss(self,y,o):
        return CF.mse(y,o)

    def __d_loss(self,y,o):
        return CF.mse_derivate(y,o)

    # funcao para selecionar a saida do neuronio
    def __selector(self,z):
        return CF.softmax(z)

    def __d_selector(self,z,alpha):
        return CF.softmax_derivate(z,alpha)

    # funcao para ativar cada neuronio
    def __activation(self,z):
        return CF.sigmoid2(z)

    def __d_activation(self,z,alpha):
        return CF.sigmoid2_derivate(z,alpha)
    
    # funcao que realiza a entrada de toda camada anterior
    def __layer(self,x,w,b):
        #return CF.dotMatrix(x,w,b)
        t = timer()
        a = CF.dotMatrix(x,w,b)
        t = timer() - t
        #print("DotMatrix ms = ", round(t * 1000, 3))
        return a

    def __d_layer(self,x,w,alpha):
        return CF.dotMatrix_derivate(x,w,alpha)
        
    def __feedForward(self,x):

        for w, b in zip(self.weights,self.biases):
            x = self.__layer(self.__activation(x),w,b)
            #x = self.__activation(self.__layer(x,w,b))

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
            nabla_b = derror # error for each bias
            derror =  self.__d_layer(z[-l-1],w,derror)

            self.weights[-l] = self.weights[-l] - (self.eta * nabla_w)
            self.biases[-l] = self.biases[-l] - (self.eta * nabla_b)
        
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

def main():
    print("ola mundo")

if __name__ == "__main__":
    main()