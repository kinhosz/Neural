acm = 0
calls = 0
avg = 0.0

def setAvg(t):
    global calls
    global acm
    global avg

    calls += 1

    if calls == 1:
        avg = t
    else:
        avg = (avg * (calls - 1) + t)/calls

def getAvg():
    global avg
    return avg

def add(t):
    global acm

    setAvg(t)

    acm += t

def get():
    global acm
    return acm

def reset():
    global acm
    global calls
    global avg

    avg = 0.0
    acm = 0
    calls = 0