def greaterIdx(l: list):
    idx = None
    score = None
    
    for i in range(len(l)):
        if not idx or score < l[i]:
            score = l[i]
            idx = i
    
    return idx