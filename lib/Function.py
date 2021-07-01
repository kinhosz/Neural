import numpy as np

def mse(predicted,target):
	n = len(predicted)
	error = (1/n) * np.sum(np.square(predicted - target))
	return error

def mse_derivate(predicted,target):
	n = len(predicted)
	return 2.0*(predicted - target)/n

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

def sigmoid2(z):
    return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)

def sigmoid2_derivate(z,alpha):
    return alpha*(2.0*np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))

def dotMatrix(x,w,b):
	return x.dot(w) + b

def dotMatrix_derivate(x,w,alpha):
	return alpha.dot(w.transpose())

def main():

	print(softmax(np.array([1.0,9.0])))

if __name__ == "__main__":
	main()