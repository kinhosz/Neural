import numpy as np

def mse_cpu(predicted, target): 
	error = np.sum(np.square(predicted - target))/2.0
	return error

def mse_derivate_cpu(predicted, target):
	return (predicted - target)

def softmax_cpu(z, *args):
	z = np.exp(z)
	sumT = z.sum()
	z = z/sumT
	return z

def softmax_derivate_cpu(z, alpha):
	soft = np.exp(z)
	S = soft.sum()
	beta = (alpha*soft).sum()/S
	soft = soft*(alpha - beta)/S
	return soft

def sigmoid2_cpu(z):
	return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)

def sigmoid2_derivate_cpu(z, alpha):
    return alpha*(2.0*np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))

def dotMatrix_cpu(x, w, b):
	return x.dot(w) + b

def dotMatrix_derivate_cpu(w, alpha):
	return alpha.dot(w.transpose())

def transposeDot_cpu(x, derror):
	return x.transpose().dot(derror)

def updateWeights_cpu(w, eta, nabla):
	w = w - (eta * nabla)
	return w