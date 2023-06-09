from .vars import transfer_register, book
from .serial import Serial
from .deallocate import deallocate
from timeit import default_timer as timer

def displayUnnusedMemo():
    unnused = 0
    total = 0
    for s in transfer_register['CPU'].values():
        total += 1
        if abs(s.timer() - timer()) > 0.5:
            unnused += 1
    
    #print("unnused memo: {} / {}".format(round(100 * unnused/total, 3), len(transfer_register['CPU'].keys())))

def serialRegister(data, mode):
    #deallocate()
    serial = Serial(data, mode)

    cpu_id, gpu_id = serial.getIds()
    transfer_register['CPU'][cpu_id] = serial
    transfer_register['GPU'][gpu_id] = serial
    #displayUnnusedMemo()

def preregister(id):
    book[id] = True

def isPermitted(id):
    return id is book.keys()