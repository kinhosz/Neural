import math
from numba import cuda, float64, float32, int64

DIM1 = (1024, )
DIM2 = (4, 256)

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

@cuda.jit
def mse(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[0]:
        diff = predicted[pos] - target[pos]
        diff = (diff * diff) / 2.0
        cuda.atomic.add(result, 0, diff)

@cuda.jit
def mse_derivate(result, predicted, target):
    pos = cuda.grid(1)

    if pos < predicted.shape[1]:
        result[0, pos] = predicted[0, pos] - target[0, pos]

@cuda.jit
def softmax_p1(arr, z, res):
	local_sum = cuda.shared.array(shape=DIM1, dtype=float64)

	x = cuda.grid(1)
	tx = cuda.threadIdx.x
	local_sum[tx] = float64(.0)

	if x >= arr.shape[1]:
		return

	val = math.exp(z[0, x])
	arr[0, x] = val
	local_sum[tx] = val
	cuda.syncthreads()

	if tx != 0:
		return
	
	for i in range(1, cuda.blockDim.x):
		local_sum[0] += local_sum[i]

	cuda.atomic.add(res, 0, local_sum[0])

@cuda.jit
def softmax_p2(arr, sumT):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = arr[0, x] / sumT[0]

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

@cuda.jit
def sigmoid2(arr, A):
	x = cuda.grid(1)

	if x < arr.shape[1]:
		arr[0, x] = 2.0 * (1.0 / (1.0 + math.exp(-A[0, x]))) - 1.0

@cuda.jit
def sigmoid2_derivate(arr, A, alpha):
	x = cuda.grid(1)

	if x < A.shape[1]:
		value = A[0, x]
		arr[0, x] = alpha[0, x] * (2.0 * math.exp(-value) / ( (1.0 + math.exp(-value)) * (1.0 + math.exp(-value))))

@cuda.jit
def sum(arr, C):
	x, y = cuda.grid(2)

	if x >= arr.shape[0] or y >= arr.shape[1]:
		return

	arr[x, y] += C[x, y]

@cuda.jit
def dotMatrix(arr, A, B):
	x, y = cuda.grid(2)
	
	temp = cuda.shared.array(shape=DIM2, dtype=float64)
	sA = cuda.shared.array(shape=(1, DIM2[0]), dtype=float64)

	tx = cuda.threadIdx.x
	ty = cuda.threadIdx.y
	temp[tx, ty] = float64(.0)

	if x >= B.shape[0] or y >= B.shape[1]:
		return
	
	if ty == 0:
		sA[0, tx] = A[0, x]
	cuda.syncthreads()

	temp[tx, ty] = sA[0, tx] * B[x, y]
	cuda.syncthreads()

	if tx != 0:
		return
	
	for i in range(1, cuda.blockDim.x):
		temp[0, ty] += temp[i, ty]
	cuda.atomic.add(arr, (0, y), temp[0, ty])

@cuda.jit
def dotMatrix_derivate(arr, w, alpha):
	x, y, z = cuda.grid(3)

	if x >= arr.shape[0] or y >= arr.shape[1] or z >= alpha.shape[1]:
		return

	cuda.atomic.add(arr, (x, y), alpha[x, z] * w[y, z])

@cuda.jit
def transposeDot(arr, A, B):
	x, y = cuda.grid(2)

	if x < arr.shape[0] and y < arr.shape[1]:
		arr[x, y] = A[0, x] * B[0, y]

@cuda.jit
def updateWeights(w, eta, nabla):
	x, y = cuda.grid(2)

	if x < w.shape[0] and y < w.shape[1]:
		w[x, y] -= eta[0] * nabla[x, y]