from numba import cuda
from timeit import default_timer as timer

def loadTo(*args, mode):
    ret_args = []
    for arg in args:
        if cuda.is_cuda_array(arg):
            if mode == 'GPU':
                ret_args.append(arg)
            else:
                ret_args.append(arg.copy_to_host())
        else:
            if mode == 'GPU':
                ret_args.append(cuda.to_device(arg))
            else:
                ret_args.append(arg)

    return ret_args


