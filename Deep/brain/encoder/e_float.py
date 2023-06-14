import struct

def fromFloat(data):
    return struct.pack('>d', data)