import numpy as np
from . import decoder, encoder

MINIMUM_SIZE = 12
SIGNATURE_SIZE = 7
INT_SIZE = 4
LAYER_HEADER_SIZE = 9
FLOAT_SIZE = 8

class Builder(object): 
    def __init__(self, path):
        self.__size = None
        self.__architecture = []
        self.__biases = []
        self.__weights = []
        self.__error = {
            0: 'type `brain` has an invalid size',
            1: 'type `brain` has not valid signature',
            2: 'data is corrupted. The general security code not match',
            3: 'layer header has no sufficient size',
            4: 'layer target has no match',
            5: 'weight has no sufficient size',
            6: 'weitght target has no match',
            7: 'biase has no sufficient size',
            8: 'biase target has no match',
            9: 'layers are not connect, shapes should be (x, y), (y, z)',
        }
        
        data = self.__read(path)
        self.__build(data)
    
    def __read(self, path):
        data = b''
        with open(path, 'rb') as file:
            data = file.read()
        
        return data
    
    def __build(self, data):
        if len(data) < MINIMUM_SIZE:
            self.__raise(0)
        
        magic_number = decoder.toInt(encoder.securityCode(data[0:-1]))
        if magic_number != data[-1]:
            self.__raise(2)
        
        prefix = 0
        signature = decoder.toString(data[prefix:prefix+SIGNATURE_SIZE])
        if signature != 'kinhosz':
            self.__raise(1)
        
        prefix += SIGNATURE_SIZE
        
        self.__size = decoder.toInt(data[prefix:prefix + INT_SIZE])
        prefix += INT_SIZE
        
        for _ in range(1, self.__size):
            prefix = self.__buildLayer(data, prefix)
        
        for i in range(1, self.__size - 1):
            if self.__weights[i].shape[0] != self.__weights[i-1].shape[1]:
                self.__raise(9)
        
        for i in range(self.__size - 1):
            self.__architecture.append(self.__weights[i].shape[0])
        self.__architecture.append(self.__weights[-1].shape[1])
    
    def __buildLayer(self, data, offset):
        if len(data) < offset + LAYER_HEADER_SIZE:
            self.__raise(3)
        
        target = chr(data[offset])
        if target != 'L':
            self.__raise(4)
        
        offset += 1
        
        dim1 = decoder.toInt(data[offset:offset+INT_SIZE])
        offset += INT_SIZE
        dim2 = decoder.toInt(data[offset:offset+INT_SIZE])
        offset +=  INT_SIZE
        
        offset = self.__buildWeight(data, dim1, dim2, offset)
        offset = self.__buildBiase(data, dim2, offset)
        
        return offset
    
    def __buildWeight(self, data, dim1, dim2, offset):
        REQUIRED_LEN = offset + FLOAT_SIZE * (dim1 * dim2) + 1
        
        if len(data) < REQUIRED_LEN:
            self.__raise(5)
        
        target = chr(data[offset])
        if target != 'W':
            self.__raise(6)
        
        offset += 1
        
        weight = np.empty((dim1, dim2))
        
        for i in range(dim1):
            for j in range(dim2):
                weight[i, j] = decoder.toFloat(data[offset:offset+FLOAT_SIZE])
                offset += FLOAT_SIZE
        
        self.__weights.append(weight)
        
        return offset
    
    def __buildBiase(self, data, dim, offset):
        REQUIRED_LEN = offset + FLOAT_SIZE * dim + 1
        
        if len(data) < REQUIRED_LEN:
            self.__raise(7)
        
        target = chr(data[offset])
        if target != 'B':
            self.__raise(8)
        
        offset += 1
        
        biase = np.empty((1, dim))
        
        for i in range(dim):
            biase[0, i] = decoder.toFloat(data[offset:offset+FLOAT_SIZE])
            offset += FLOAT_SIZE
        
        self.__biases.append(biase)
        
        return offset
    
    def __raise(self, flag):
        raise TypeError(self.__error[flag])
    
    def size(self):
        return self.__size
    
    def architecture(self):
        return self.__architecture
    
    def biases(self):
        return self.__biases
    
    def weights(self):
        return self.__weights