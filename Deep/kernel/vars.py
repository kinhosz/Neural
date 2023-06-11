acm = 0
calls = 0
avg = 0.0

def add(t):
    global acm

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