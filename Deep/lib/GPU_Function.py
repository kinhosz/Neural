import numpy as np
import math
from numba import cuda
from timeit import default_timer as timer

THREADSPERBLOCK = 1024
MINIMUMBLOCKSIZE = 28
EPS = 1e-10

def config():
	#np.seterr(all="none")
	pass

def mse_cpu(predicted,target): 
	error = np.sum(np.square(predicted - target))/2.0
	return error

@cuda.jit
def mse(predicted, target, result):
    pos = cuda.grid(1)

    if pos < len(predicted):
        diff = predicted[pos] - target[pos]
        diff = (diff * diff) / 2.0
        cuda.atomic.add(result, 0, diff)

def mse_derivate_cpu(predicted,target):
	return (predicted - target)

@cuda.jit
def mse_derivate(predicted, target, result):
    pos = cuda.grid(1)

    if pos < len(predicted):
        result[pos] = predicted[pos] - target[pos]

def relu(z):
	l = len(z)
	for i in range(0,l):
		if z[i][0] < 0.0:
			z[i][0] = 0.0

	return z

def relu_derivate(z,alpha):
	l = len(z)
	for i in range(0,l):
		if z[i][0] <= 0.0:
			z[i][0] = 0.0
		else:
			z[i][0] = 1.0

	z = z*alpha
	return z

def relu2(z):
	d = 0.01
	l = len(z)
	for i in range(0,l):
		if z[i][0] < 0.0:
			z[i][0] = z[i][0]*d

	return z

def relu2_derivate(z,alpha):
	d = 0.01
	l = len(z)
	for i in range(0,l):
		if z[i][0] <= 0.0:
			z[i][0] = d
		else:
			z[i][0] = 1.0

	z = z*alpha
	return z 

def softmax(z):
	z = np.exp(z)
	sumT = z.sum()
	z = z/sumT
	return z

def softmax_derivate(z,alpha):
	soft = np.exp(z)
	S = soft.sum()
	beta = (alpha*soft).sum()/S
	soft = soft*(alpha - beta)/S
	return soft

def sigmoid(z):
	return 20.0*(1.0/(1.0 + np.exp(-z))) - 10 # (-10,10)

def sigmoid_derivate(z,alpha):
    return alpha*(np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))*20.0

def sigmoid2_teste(z):
    #return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)
    l = list(z[0])

    aux = []

    for i in l:
    	try:
    		i = math.exp(-i) + 1.0
    	except OverflowError:
    		i = float('inf')
    	
    	i = 2.0*(1.0/i) - 1.0
    	aux.append(i)

    return np.array([aux])

def sigmoid2(z):
	return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)

def sigmoid2_derivate(z,alpha):
    return alpha*(2.0*np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))

def dotMatrix(x,w,b):
	return x.dot(w) + b

def dotMatrix_derivate(x,w,alpha):
	return alpha.dot(w.transpose())

# unit tests
def mse_test():
    LEN_ARRAY = 5000

    stream = cuda.stream()

    predicted = np.random.randn(LEN_ARRAY)
    target = np.random.randn(LEN_ARRAY)
    result = np.zeros(1)

    d_predicted = cuda.to_device(predicted, stream=stream)
    d_target = cuda.to_device(target, stream=stream)
    d_result = cuda.to_device(result, stream=stream)

    cpu_answer = mse_cpu(predicted, target)

    BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK)

    mse[BLOCKSPERGRID, THREADSPERBLOCK](d_predicted, d_target, d_result)

    h_result = d_result.copy_to_host(stream=stream)

    assert abs(h_result[0] - cpu_answer) <= EPS

def mse_derivate_test():
    LEN_ARRAY = 200000

    stream = cuda.stream()

    predicted = np.random.randn(LEN_ARRAY)
    target = np.random.randn(LEN_ARRAY)
    result = np.random.randn(LEN_ARRAY)

    d_predicted = cuda.to_device(predicted, stream=stream)
    d_target = cuda.to_device(target, stream=stream)
    d_result = cuda.to_device(result, stream=stream)

    r_cpu = mse_derivate_cpu(predicted, target)
    
    needblocksgrid = (LEN_ARRAY + THREADSPERBLOCK - 1) // THREADSPERBLOCK
    BLOCKSPERGRID = max(MINIMUMBLOCKSIZE, needblocksgrid)

    mse_derivate[BLOCKSPERGRID, THREADSPERBLOCK](d_predicted, d_target, d_result)

    r_gpu = d_result.copy_to_host(stream=stream)

    for i in range(len(r_cpu)):
        assert abs(r_cpu[i] - r_gpu[i]) <= EPS

def test():
    mse_test()
    mse_derivate_test()

if __name__ == "__main__":
	test()