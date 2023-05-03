import numpy as np
from numba import cuda
from Deep.lib.GPU import *
from .utils import *

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

def dotMatrix_test():
	LEN_ARRAY = 100000
	LEN_ARRAY1 = int(math.sqrt(LEN_ARRAY))
	LEN_ARRAY2 = LEN_ARRAY1
	
	A_host = np.random.randn(1, LEN_ARRAY1)
	B_host = np.random.randn(LEN_ARRAY1, LEN_ARRAY2)
	C_host = np.random.randn(1, LEN_ARRAY2)
	
	A_device = cuda.to_device(A_host)
	B_device = cuda.to_device(B_host)
	C_device = cuda.to_device(C_host)
	
	arr_host = np.random.randn(1, LEN_ARRAY2)
	arr_device = cuda.to_device(arr_host)
	
	ans = dotMatrix_cpu(A_host, B_host, C_host)
	
	dotMatrix[kernelConfig2D(1, LEN_ARRAY2)](arr_device, A_device, B_device, C_device)
	cuda.synchronize()
	
	arr_host = arr_device.copy_to_host()
	
	for i in range(LEN_ARRAY2):
		if abs(ans[0, i] - arr_host[0, i]) > EPS:
			return False
	
	return True

def test():
	tests = [memset_test, memset2_test, mse_test, mse_derivate_test, softmax_test,
	  		softmax_derivate_test, sigmoid2_test, sigmoid2_derivate_test, copy_test,
			dotMatrix_test, dotMatrix_derivate_test, transposeDot_test, updateWeights_test]

	gerador = logger("correctness", tests)
	while gerador != None:
		gerador = next(gerador)

if __name__ == "__main__":
	test()