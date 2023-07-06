from numba.cuda.cudadrv.devicearray import DeviceNDArray
from .kernel import inverse_matrix_multiplication, memset_arr_1

def dot_matrix_derivate(const_matrix: DeviceNDArray, alphas: DeviceNDArray, buffer: DeviceNDArray) -> DeviceNDArray:
    """alpha.dot(const_matrix.transpose())

    Args:
        const_matrix (DeviceNDArray): [N][M]
        alphas (DeviceNDArray): [BATCH][1][M]
        buffer (DeviceNDArray): [BATCH][1][N]
    """
    
    memset_arr_1(buffer)
    inverse_matrix_multiplication(buffer=buffer, const_matrix=const_matrix, alphas=alphas)
    
    return buffer
