import struct

def toFloat(data):
    return struct.unpack('>d', data)[0]