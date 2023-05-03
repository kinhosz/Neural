import numpy as np
import math
from numba import cuda, float64
from timeit import default_timer as timer
from colorama import Fore, init

MINIMUMBLOCKSIZE = 28
EPS = 1e-10
THREADSPERBLOCK = 1024

@cuda.jit
def copy(arr, A):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = A[0, x]

@cuda.jit
def memset(arr):
	x = cuda.grid(1)

	if x < arr.shape[0]:
		arr[x] = float64(0.)

@cuda.jit
def memset2(arr):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = float64(0.)

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

def softmax_derivate_cpu(z, alpha):
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
def sigmoid2_derivate(arr, A, alpha):
	x = cuda.grid(1)

	if x < A.shape[1]:
		value = A[0, x]
		arr[0, x] = alpha[0, x] * (2.0 * math.exp(-value) / ( (1.0 + math.exp(-value)) * (1.0 + math.exp(-value))))

def dotMatrix_cpu(x, w, b):
	return x.dot(w) + b

@cuda.jit
def sum(arr, C):
	x, y = cuda.grid(2)

	if x >= arr.shape[0] or y >= arr.shape[1]:
		return

	arr[x, y] += C[x, y]

@cuda.jit
def dotMatrix(arr, A, B):
	x, y, z = cuda.grid(3)

	if x >= A.shape[0] or y >= A.shape[1] or z >= B.shape[1]:
		return
	
	cuda.atomic.add(arr, (x, z), A[x, y] * B[y, z])

def dotMatrix_derivate_cpu(w, alpha):
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
			if t[i] >= sz[i] or threads == 1:
				continue
			threads //= 2
			t[i] *= 2
			upd = True
	
	blockspergrid_x = ceil(size_x, t[0])
	blockspergrid_y = ceil(size_y, t[1])

	blocks = blockspergrid_x * blockspergrid_y
	if blocks < MINIMUMBLOCKSIZE:
		add = MINIMUMBLOCKSIZE - blocks
		blocks //= blockspergrid_y
		blockspergrid_y += ceil(add, blocks)
	
	blockspergrid = (blockspergrid_x, blockspergrid_y)
	threadsperblock = (t[0], t[1])

	return (blockspergrid, threadsperblock)

def kernelConfig3D(size_x, size_y, size_z):
	threads = THREADSPERBLOCK

	sz = [size_x, size_y, size_z]
	t = [1, 1, 1]

	upd = True
	while threads > 1 and upd:
		upd = False
		for i in range(3):
			if t[i] >= sz[i] or threads == 1:
				continue
			threads //= 2
			t[i] *= 2
			upd = True
	
	blockspergrid_x = ceil(size_x, t[0])
	blockspergrid_y = ceil(size_y, t[1])
	blockspergrid_z = ceil(size_z, t[2])

	blocks = blockspergrid_x * blockspergrid_y * blockspergrid_z
	if blocks < MINIMUMBLOCKSIZE:
		add = MINIMUMBLOCKSIZE - blocks
		blocks //= blockspergrid_z
		blockspergrid_z += ceil(add, blocks)
	
	blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
	threadsperblock = (t[0], t[1], t[2])

	return (blockspergrid, threadsperblock)

def memset_test():
	LEN_ARRAY = 2000

	arr_host = np.random.randn(LEN_ARRAY)
	arr_device = cuda.to_device(arr_host)

	memset[kernelConfig1D(LEN_ARRAY)](arr_device)
	cuda.synchronize()

	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		if abs(arr_host[i]) > EPS:
			return False
	
	return True

def memset2_test():
	LEN_ARRAY1 = 2000
	LEN_ARRAY2 = 3000

	arr_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	arr_device = cuda.to_device(arr_host)

	memset2[kernelConfig2D(LEN_ARRAY1, LEN_ARRAY2)](arr_device)

	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			if abs(arr_host[i, j]) > EPS:
				return False

	return True

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

	if abs(arr_cpu - arr_gpu) > EPS:
		return False

	return True

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
		if abs(arr_cpu[0, i] - arr_gpu[0, i]) > EPS:
			return False
	
	return True

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
		if abs(z_cpu[0, i] - arr_gpu[0, i]) > EPS:
			return False
	
	return True

def softmax_derivate_test():
	LEN_ARRAY = 2000

	z_host = np.random.randn(1, LEN_ARRAY)
	alpha_host = np.random.randn(1, LEN_ARRAY)
	simple_sum_host = np.zeros(1)
	sum_times_alpha_host = np.zeros(1)
	arr_gpu = np.random.randn(1, LEN_ARRAY)

	z_device = cuda.to_device(z_host)
	alpha_device = cuda.to_device(alpha_host)
	simple_sum_device = cuda.to_device(simple_sum_host)
	sum_times_alpha_device = cuda.to_device(sum_times_alpha_host)
	arr_device = cuda.to_device(arr_gpu)

	z_cpu = softmax_derivate_cpu(z_host, alpha_host)

	softmax_sum_derivate[kernelConfig1D(LEN_ARRAY)](arr_device, z_device, alpha_device, simple_sum_device, sum_times_alpha_device)
	cuda.synchronize()
	softmax_derivate[kernelConfig1D(LEN_ARRAY)](arr_device, alpha_device, simple_sum_device, sum_times_alpha_device)
	cuda.synchronize()

	arr_gpu = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		if abs(arr_gpu[0, i] - z_cpu[0, i]) > EPS:
			return False
	
	return True

def sigmoid2_test():
	LEN_ARRAY = 2000

	A_host = np.random.randn(1, LEN_ARRAY)
	A_device = cuda.to_device(A_host)
	arr_gpu = np.random.randn(1, LEN_ARRAY)
	arr_device = cuda.to_device(arr_gpu)

	z_cpu = sigmoid2_cpu(A_host)

	sigmoid2[kernelConfig1D(LEN_ARRAY)](arr_device, A_device)
	arr_gpu = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		if abs(z_cpu[0, i] - arr_gpu[0, i]) > EPS:
			return False
	
	return True

def sigmoid2_derivate_test():
	LEN_ARRAY = 2000

	A_host = np.random.randn(1, LEN_ARRAY)
	A_device = cuda.to_device(A_host)
	alpha_host = np.random.randn(1, LEN_ARRAY)
	alpha_device = cuda.to_device(alpha_host)
	arr_gpu = np.random.randn(1, LEN_ARRAY)
	arr_device = cuda.to_device(arr_gpu)

	z_cpu = sigmoid2_derivate_cpu(A_host, alpha_host)

	sigmoid2_derivate[kernelConfig1D(LEN_ARRAY)](arr_device, A_device, alpha_device)
	arr_gpu = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		if abs(z_cpu[0, i] - arr_gpu[0, i]) > EPS:
			return False
	
	return True

def copy_test():
	LEN_ARRAY = 2000

	A_host = np.random.randn(1, LEN_ARRAY)
	arr_host = np.random.randn(1, LEN_ARRAY)

	A_device = cuda.to_device(A_host)
	arr_device = cuda.to_device(arr_host)

	copy[kernelConfig1D(LEN_ARRAY)](arr_device, A_device)

	A_host = A_device.copy_to_host()
	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY):
		if abs(A_host[0, i] - arr_host[0, i]) > EPS:
			return False
	
	return True

def dotMatrix_loop_test():
	LEN_ARRAY1 = 200

	A_host = np.ones([1, LEN_ARRAY1], dtype=np.float64)
	B_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY1)
	C_host = np.random.randn(1, LEN_ARRAY1)

	A_device = cuda.to_device(A_host)
	B_device = cuda.to_device(B_host)
	C_device = cuda.to_device(C_host)

	arr_host = np.random.randn(1, LEN_ARRAY1)
	arr_device = cuda.to_device(arr_host)

	LOOP = 200

	for i in range(LOOP):
		A_host = dotMatrix_cpu(A_host, B_host, C_host)
	
	for i in range(LOOP):
		memset2[kernelConfig2D(1, LEN_ARRAY1)](arr_device)
		cuda.synchronize()
		dotMatrix[kernelConfig3D(1, LEN_ARRAY1, LEN_ARRAY1)](arr_device, A_device, B_device)
		cuda.synchronize()
		sum[kernelConfig2D(1, LEN_ARRAY1)](arr_device, C_device)
		cuda.synchronize()
		copy[kernelConfig1D(LEN_ARRAY1)](A_device, arr_device)
		cuda.synchronize()
	
	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		if (A_host[0, i] < EPS and arr_host[0, i] > EPS) or (A_host[0, i] > EPS and arr_host[0, i] < EPS):
			return False
		if abs(np.log(abs(A_host[0, i])) - np.log(abs(arr_host[0, i]))) > EPS:
			return False
	
	return True

def dotMatrix_test():
	LEN_ARRAY1 = 200

	A_host = np.ones([1, LEN_ARRAY1], dtype=np.float64)
	B_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY1)
	C_host = np.random.randn(1, LEN_ARRAY1)

	A_device = cuda.to_device(A_host)
	B_device = cuda.to_device(B_host)
	C_device = cuda.to_device(C_host)

	arr_host = np.random.randn(1, LEN_ARRAY1)
	arr_device = cuda.to_device(arr_host)

	LOOP = 1

	for i in range(LOOP):
		A_host = dotMatrix_cpu(A_host, B_host, C_host)
	
	for i in range(LOOP):
		memset2[kernelConfig2D(1, LEN_ARRAY1)](arr_device)
		cuda.synchronize()
		dotMatrix[kernelConfig3D(1, LEN_ARRAY1, LEN_ARRAY1)](arr_device, A_device, B_device)
		cuda.synchronize()
		sum[kernelConfig2D(1, LEN_ARRAY1)](arr_device, C_device)
		cuda.synchronize()
		copy[kernelConfig1D(LEN_ARRAY1)](A_device, arr_device)
		cuda.synchronize()
	
	arr_host = arr_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		if abs(abs(A_host[0, i]) - abs(arr_host[0, i])) > EPS:
			return False
	
	return True

def dotMatrix_derivate_test():
	LEN_ARRAY1 = 200
	LEN_ARRAY2 = 100

	w_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	alpha_host = np.random.randn(1, LEN_ARRAY2)
	aux = np.random.randn(1, LEN_ARRAY1)
	w_device = cuda.to_device(w_host)
	alpha_device = cuda.to_device(alpha_host)
	ans_device = cuda.to_device(aux)

	ans_cpu = dotMatrix_derivate_cpu(w_host, alpha_host)
	memset2[kernelConfig2D(1, LEN_ARRAY1)](ans_device)
	dotMatrix_derivate[kernelConfig3D(1, LEN_ARRAY1, LEN_ARRAY2)](ans_device, w_device, alpha_device)
	cuda.synchronize()
	ans_gpu = ans_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		if abs(ans_cpu[0, i] - ans_gpu[0, i]) > EPS:
			return False
	
	return True

def transposeDot_test():
	LEN_ARRAY1 = 100
	LEN_ARRAY2 = 200

	x_host = np.random.randn(1, LEN_ARRAY1)
	derror_host = np.random.randn(1, LEN_ARRAY2)
	x_device = cuda.to_device(x_host)
	derror_device = cuda.to_device(derror_host)
	ans_cpu = transposeDot_cpu(x_host, derror_host)
	ans = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	ans_device = cuda.to_device(ans)
	transposeDot[kernelConfig2D(LEN_ARRAY1, LEN_ARRAY2)](ans_device, x_device, derror_device)
	ans_gpu = ans_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			if abs(ans_gpu[i, j] - ans_cpu[i, j]) > EPS:
				return False
	
	return True

def updateWeights_test():
	LEN_ARRAY1 = 100
	LEN_ARRAY2 = 200

	w_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	eta_host = np.random.randn(1)
	nabla_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	w_device = cuda.to_device(w_host)
	eta_device = cuda.to_device(eta_host)
	nabla_device = cuda.to_device(nabla_host)
	ans_cpu = updateWeights_cpu(w_host, eta_host, nabla_host)
	updateWeights[kernelConfig2D(LEN_ARRAY1, LEN_ARRAY2)](w_device, eta_device, nabla_device)
	ans_gpu = w_device.copy_to_host()

	for i in range(LEN_ARRAY1):
		for j in range(LEN_ARRAY2):
			if abs(ans_cpu[i, j] - ans_gpu[i, j]) > EPS:
				return False
	
	return True

def test():
	init()
	tests = [memset_test, memset2_test, mse_test, mse_derivate_test, softmax_test,
	  		softmax_derivate_test, sigmoid2_test, sigmoid2_derivate_test, copy_test,
			dotMatrix_loop_test, dotMatrix_test, dotMatrix_derivate_test,
			transposeDot_test, updateWeights_test]

	failed_tests = []

	print(Fore.YELLOW + 'Tests started')
	for currentTest in tests:
		ok = currentTest()

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
	test()