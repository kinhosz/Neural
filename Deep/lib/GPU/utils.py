MINIMUMBLOCKSIZE = 28
EPS = 1e-10
THREADSPERBLOCK = 1024

def ceil(A, B):
	return (A + B - 1) // B

def kernelConfig1D(size_x):
	threads_x = THREADSPERBLOCK
	blockspergrid_x = max(ceil(size_x, threads_x), MINIMUMBLOCKSIZE)

	return (blockspergrid_x, threads_x)

def kernelConfig2D(size_x, size_y, shape=None):
	threads = THREADSPERBLOCK

	sz = [size_x, size_y]
	t = [1, 1]

	t[0] = 32
	t[1] = 32

	if shape:
		t[0] = shape[0]
		t[1] = shape[1]
	
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
