from . import encoder

class Wrapper(object):
    def __init__(self, layers):
        self.__data = b''
        self.__error = {
            0: 'weight has shape different of (x, y)',
            1: 'biase has shape different of (1, x)',
            2: 'weight has second dimension different of biase. Should be: (x, y), (1, y)',
            3: 'values in weight are not float',
            4: 'values in biase are not float',
            5: 'layers dimensions are not connected. Should be: (x, y), (y, z)',
        }
        
        self.__checkType(layers)
        self.__buildData(layers)
    
    def __raise(self, flag):
        raise TypeError(self.__error[flag])
    
    def __checkType(self, layers):
        for w, b in layers:
            if len(w.shape) != 2:
                self.__raise(0)
            if len(b.shape) != 2:
                self.__raise(1)
            
            if w.shape[1] != b.shape[1]:
                self.__raise(2)
            
            for w_x in w:
                for w_x_y in w_x:
                    if not isinstance(w_x_y, float):
                        self.__raise(3)
            
            for b_x in b[0]:
                if not isinstance(b_x, float):
                    self.__raise(4)
        
        for i in range(1, len(layers)):
            if layers[i][0].shape[0] != layers[i-1][0].shape[1]:
                self.__raise(5)
    
    def __buildData(self, layers):
        header = b''
        header += encoder.fromString('kinhosz')
        header += encoder.fromInt(len(layers) + 1)
        
        body = b''
        
        for w, b in layers:
            layer = b''
            layer += encoder.fromString('L')
            layer += encoder.fromInt(w.shape[0])
            layer += encoder.fromInt(w.shape[1])

            layer += encoder.fromString('W')
            for w_x in w:
                for w_x_y in w_x:
                    layer += encoder.fromFloat(w_x_y)
            
            layer += encoder.fromString('B')
            for b_x in b[0]:
                layer += encoder.fromFloat(b_x)
            
            body += layer
        
        data = header + body
        
        security_code = encoder.securityCode(data)
        
        data += security_code
        self.__data = data
    
    def data(self):
        return self.__data

