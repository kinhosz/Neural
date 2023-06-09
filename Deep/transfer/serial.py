from numba import cuda
from timeit import default_timer as timer

class Serial(object):
    def __init__(self, data, mode):
        self.__toGPUStream = cuda.stream()
        self.__toCPUStream = cuda.stream()
        self.__cpuData = None
        self.__gpuData = None
        self.__timer = timer()

        if mode == 'GPU':
            self.__gpuData = data
            self.__cpuData = data.copy_to_host(stream=self.__toCPUStream)
        elif mode == 'CPU':
            self.__cpuData = data
            self.__gpuData = cuda.to_device(data, stream=self.__toGPUStream)
    
    def __getFromCPU(self):
        self.__toCPUStream.synchronize()
        return self.__cpuData
    
    def __getFromGPU(self):
        self.__toGPUStream.synchronize()
        return self.__gpuData

    def getFrom(self, mode):
        if mode == 'CPU':
            return self.__getFromCPU()
        elif mode == 'GPU':
            return self.__getFromGPU()

    def getIds(self):
        return id(self.__cpuData), id(self.__gpuData)
    
    def timer(self):
        return self.__timer
