from numba import cuda
from timeit import default_timer as timer

def loadTo(*args, mode):
    ret_args = []
    for arg in args:
        if not cuda.is_cuda_array(arg):
            ret_args.append(cuda.to_device(arg))
        else:
            ret_args.append(arg)

    return ret_args


