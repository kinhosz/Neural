import numpy as np
import random
import math
from lib import Function as F

class Network(object):

    def __init__(self,sizes,eta=0.01):
        self.eta = eta
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(1,x) for x in sizes[1:]]

    ############################################################################
    ############# NETWORK ######################################################

    # funcao para calcular o erro da rede
    # y = minha saida
    # o = output esperado
    def loss(self,y,o):
        return F.mse(y,o)

    def d_loss(self,y,o):
        return F.mse_derivate(y,o)

    # funcao para selecionar a saida do neuronio
    def selector(self,z):
        return F.softmax(z)

    def d_selector(self,z,alpha):
        return F.softmax_derivate(z,alpha)

    # funcao para ativar cada neuronio
    def activation(self,z):
        return F.sigmoid2(z)

    def d_activation(self,z,alpha):
        return F.sigmoid2_derivate(z,alpha)
    
    # funcao que realiza a entrada de toda camada anterior
    def layer(self,x,w,b):
        return F.dotMatrix(x,w,b)

    def d_layer(self,x,w,alpha):
        return F.dotMatrix_derivate(x,w,alpha)
        
    def feedForward(self,x):

        for w, b in zip(self.weights,self.biases):
            x = self.activation(self.layer(x,w,b))

        return self.selector(x)

    def backPropagation(self,x,target):

        # feedForward
        z = [x] # save all Zs
        activations = [] # save all activations

        for w, b in zip(self.weights,self.biases):
            x = self.layer(x,w,b)
            activations.append(x)
            x = self.activation(x)
            z.append(x)

        y = self.selector(x)

        derror = self.d_loss(y,target)
        derror = self.d_selector(z[self.num_layers - 1],derror)

        for l in range(1, self.num_layers):
            w = self.weights[-l]
            b = self.biases[-l]

            derror = self.d_activation(activations[-l],derror)
            nabla_w = z[-l-1].transpose().dot(derror) # error for each wij
            nabla_b = derror # error for each bias
            derror =  self.d_layer(z[-l-1],w,derror)

            self.weights[-l] = self.weights[-l] - (self.eta * nabla_w)
            self.biases[-l] = self.biases[-l] - (self.eta * nabla_b)

    def send(self, l):
        x = self.activation(np.array([l]))
        return self.feedForward(x)[0]

    def learn(self,x,y):
        x = self.activation(np.array([x]))
        y = np.array([y])
        self.backPropagation(x,y)

def main():
    print("ola mundo")

if __name__ == "__main__":
    main()