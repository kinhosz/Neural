def genXor(data):
    res = 0
    for i in range(len(data)):
        res ^= data[i]
    
    return res.to_bytes(1, 'big')