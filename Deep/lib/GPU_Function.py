import numpy as np
import math
from numba import cuda
from timeit import default_timer as timer
from colorama import Fore, init

MINIMUMBLOCKSIZE = 28
EPS = 1e-10
THREADSPERBLOCK = 1024

@cuda.jit
def memset(arr):
	x = cuda.grid(1)

	if x < arr.shape[0]:
		arr[x] = 0

@cuda.jit
def memset2(arr):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = 0

def mse_cpu(predicted, target): 
	error = np.sum(np.square(predicted - target))/2.0
	return error

@cuda.jit
def mse(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[0]:
        diff = predicted[pos] - target[pos]
        diff = (diff * diff) / 2.0
        cuda.atomic.add(result, 0, diff)

def mse_derivate_cpu(predicted,target):
	return (predicted - target)

@cuda.jit
def mse_derivate(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[1]:
        result[0, pos] = predicted[0, pos] - target[0, pos]

def softmax_cpu(z):
	z = np.exp(z)
	sumT = z.sum()
	z = z/sumT
	return z

@cuda.jit
def softmax_p1(arr, z, res):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = math.exp(z[0, x])
		cuda.atomic.add(res, 0, arr[0, x])

@cuda.jit
def softmax_p2(arr, sumT):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = arr[0, x] / sumT[0]

###################
def softmax_derivate_cpu(z,alpha):
	soft = np.exp(z)
	S = soft.sum()
	beta = (alpha*soft).sum()/S
	soft = soft*(alpha - beta)/S
	return soft

@cuda.jit
def softmax_sum_derivate(arr, z, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		value = z[0, x]
		value = math.exp(value)
		arr[0, x] = value
		cuda.atomic.add(simple_sum, 0, value)
		cuda.atomic.add(sum_times_alpha, 0, value * alpha[0, x])

@cuda.jit
def softmax_derivate(arr, alpha, simple_sum, sum_times_alpha):
	x = cuda.grid(1)
 
	if x < arr.shape[1]:
		value = arr[0, x]
		arr[0, x] = (value * (alpha[0, x] - (sum_times_alpha[0] / simple_sum[0]))) / simple_sum[0]

def sigmoid2_cpu(z):
	return 2.0*(1.0/(1.0 + np.exp(-z))) - 1.0 # (-1,1)

@cuda.jit
def sigmoid2(arr, A):
	x = cuda.grid(1)

	if x < arr.shape[1] and 0 < arr.shape[0]:
		arr[0, x] = 2.0 * (1.0 / (1.0 + math.exp(-A[0, x]))) - 1.0

def sigmoid2_derivate_cpu(z,alpha):
    return alpha*(2.0*np.exp(-z)/((1.0 + np.exp(-z))*(1.0 + np.exp(-z))))

@cuda.jit
def sigmoid2_derivate(arr, alpha):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		value = arr[0, x]
		arr[0, x] = alpha[0, x] * (2.0 * math.exp(-value) / ( (1.0 + math.exp(-value)) * (1.0 + math.exp(-value))))

def dotMatrix_cpu(x,w,b):
	return x.dot(w) + b

@cuda.jit
def sum(arr, C):
	x, y = cuda.grid(2)

	if x >= arr.shape[0] or y >= arr.shape[1]:
		return

	arr[x, y] += C[x, y]

@cuda.jit
def dotMatrixV3(arr, A, B):
	x, y, z = cuda.grid(3)

	if x >= A.shape[0] or y >= A.shape[1] or z >= B.shape[1]:
		return
	
	cuda.atomic.add(arr, (x, z), A[x, y] * B[y, z])
	#arr[x, z] += A[x, y] * B[y, z] # atomic.add is correct version

def dotMatrix_derivate_cpu(x,w,alpha):
	return alpha.dot(w.transpose())

@cuda.jit
def dotMatrix_derivate(arr, w, alpha):
	x, y, z = cuda.grid(3)

	if x >= arr.shape[0] or y >= arr.shape[1] or z >= alpha.shape[1]:
		return

	cuda.atomic.add(arr, (x, y), alpha[x, z] * w[y, z])

def transposeDot_cpu(x, derror):
	return x.transpose().dot(derror)

@cuda.jit
def transposeDot(arr, A, B):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = A[0, x] * B[0, y]

def updateWeights_cpu(w, eta, nabla):
	w = w - (eta * nabla)
	return w

@cuda.jit
def updateWeights(w, eta, nabla):
	x, y = cuda.grid(2)

	if x < w.shape[0] and y < w.shape[1]:
		w[x, y] -= eta[0] * nabla[x, y]

'''
Test Correctness
'''

def ceil(A, B):
	return (A + B - 1) // B

def kernelConfig1D(size_x):
	threads_x = THREADSPERBLOCK
	blockspergrid_x = max(ceil(size_x, threads_x), MINIMUMBLOCKSIZE)

	return (blockspergrid_x, threads_x)

def kernelConfig2D(size_x, size_y):
	threads = THREADSPERBLOCK

	sz = [size_x, size_y]
	t = [1, 1]

	upd = True
	while threads > 1 and upd:
		upd = False
		for i in range(2):
			if t[i] >= sz[i]:
				continue
			threads //= 2
			t[i] *= 2
			upd = True
	
	blockspergrid_x = ceil(size_x, t[0])
	blockspergrid_y = ceil(size_y, t[1])

	blocks = blockspergrid_x * blockspergrid_y
	if blocks < MINIMUMBLOCKSIZE:
		add = MINIMUMBLOCKSIZE - blocks
		blocks /= blockspergrid_y
		blockspergrid_y += ceil(add, blocks)
	
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	threadsperblock = (t[0], t[1])

	return (blockspergrid, threadsperblock)

def memset_test():
	LEN_ARRAY = 2000

	arr_host = np.random.randn(LEN_ARRAY)
	arr_device = cuda.to_device(arr_host)

	memset[kernelConfig1D(LEN_ARRAY)](arr_device)
	cuda.synchronize()

	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		assert abs(arr_host[i]) <= EPS

def memset2_test():
	LEN_ARRAY1 = 2000
	LEN_ARRAY2 = 3000

	arr_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	arr_device = cuda.to_device(arr_host)

	memset2[kernelConfig2D(LEN_ARRAY1, LEN_ARRAY2)](arr_device)

	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			assert abs(arr_host[i, j]) <= EPS

def mse_test():
	LEN_ARRAY = 2000

	predicted_host = np.random.randn(LEN_ARRAY)
	target_host = np.random.randn(LEN_ARRAY)

	predicted_device = cuda.to_device(predicted_host)
	target_device = cuda.to_device(target_host)

	arr_cpu = mse_cpu(predicted_host, target_host)

	arr_gpu = np.zeros(1)
	arr_device = cuda.to_device(arr_gpu)

	mse[kernelConfig1D(LEN_ARRAY)](arr_device, predicted_device, target_device)
	arr_gpu = arr_device.copy_to_host()

	assert abs(arr_cpu - arr_gpu) <= EPS

def mse_derivate_test():
	LEN_ARRAY = 2000

	predicted_host = np.random.randn(1, LEN_ARRAY)
	target_host = np.random.randn(1, LEN_ARRAY)

	predicted_device = cuda.to_device(predicted_host)
	target_device = cuda.to_device(target_host)

	arr_cpu = mse_derivate_cpu(predicted_host, target_host)

	arr_gpu = np.random.randn(1, LEN_ARRAY)
	arr_device = cuda.to_device(arr_gpu)

	mse_derivate[kernelConfig1D(LEN_ARRAY)](arr_device, predicted_device, target_device)
	
	arr_gpu = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		assert abs(arr_cpu[0, i] - arr_gpu[0, i]) <= EPS

def softmax_test():
	LEN_ARRAY = 2000

	z_host = np.random.randn(1, LEN_ARRAY)
	z_device = cuda.to_device(z_host)
	res_host = np.zeros(1)
	res_device = cuda.to_device(res_host)

	arr_gpu = np.random.randn(1, LEN_ARRAY)
	arr_device = cuda.to_device(arr_gpu)

	z_cpu = softmax_cpu(z_host)

	softmax_p1[kernelConfig1D(LEN_ARRAY)](arr_device, z_device, res_device)
	cuda.synchronize()
	softmax_p2[kernelConfig1D(LEN_ARRAY)](arr_device, res_device)
	cuda.synchronize()
	
	arr_gpu = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		assert abs(z_cpu[0, i] - arr_gpu[0, i]) <= EPS

def softmax_derivate_test():
	pass

def test():
	init()
	tests = [memset_test, memset2_test, mse_test, mse_derivate_test, softmax_test,
	  		softmax_derivate_test]

	failed_tests = []

	print(Fore.YELLOW + 'Tests started')
	for currentTest in tests:
		ok = True
		try:
			currentTest()
		except:
			ok = False
		status = "{}".format('.' if ok else 'F')
		
		if ok == False:
			failed_tests.append(currentTest.__name__)

		if ok:
			print(Fore.GREEN + "{}".format(status), end='')
		else:
			print(Fore.RED + "{}".format(status), end='')

	print("\n")

	if len(failed_tests) == 0:
		print(Fore.GREEN + "All tests have passed")
	else:
		print(Fore.RED + "Failed tests:")
		for t in failed_tests:
			print(Fore.BLUE + "{}".format(t))


if __name__ == "__main__":
	softmax_derivate_test()
	test()