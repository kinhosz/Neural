from .common.dataset import dataset
from .common.greaterIdx import greaterIdx
from .common.densityArr import densityArr

class Shared(object):
    def __init__(self):
        self.__images = None
        self.__labels = None
        
        self.__build()
    
    def __build(self):
        images, labels = dataset()
        self.__images = images
        self.__labels = labels
    
    def images(self):
        return self.__images
    
    def labels(self):
        return self.__labels
    
    def greaterIdx(self, l: list):
        return greaterIdx(l)
    
    def densityArr(self, idx: int, len: int):
        return densityArr(idx, len)
