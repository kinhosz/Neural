from .vars import transfer_register
from timeit import default_timer as timer

BASE_LIMIT = 100
TIME_IN_S = 0.5

def byLastAcess(e):
    return e[2]

def deallocate():
    if len(transfer_register['CPU'].keys()) < BASE_LIMIT:
        return None
    
    refs = []
    for serial_data in transfer_register['CPU'].values():
        cpu_id, gpu_id = serial_data.getIds()
        lastAcess = serial_data.timer()
        refs.append((cpu_id, gpu_id, lastAcess))
    
    refs.sort(key=byLastAcess)

    minimum_life_required = timer() - TIME_IN_S

    for ref in refs:
        if ref[2] > minimum_life_required:
            break

        cpu_id = ref[0]
        gpu_id = ref[1]
        serial = transfer_register['CPU'][cpu_id]
        del transfer_register['CPU'][cpu_id]
        del transfer_register['GPU'][gpu_id]
        del serial
