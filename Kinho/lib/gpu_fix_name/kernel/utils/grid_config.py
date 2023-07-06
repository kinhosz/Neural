from .ceil import ceil

MINIMUM_BLOCK_SIZE = 28

def grid_1(shape):
    if shape[0] >= MINIMUM_BLOCK_SIZE:
        return shape

    block_x = shape[0] + (MINIMUM_BLOCK_SIZE - shape[0])

    return (block_x, )

def grid_2(shape):
    blocks = shape[0] * shape[1]
    
    if blocks >= MINIMUM_BLOCK_SIZE:
        return shape
    
    block_x = shape[0] + ceil(MINIMUM_BLOCK_SIZE - blocks, shape[1])
    
    return (block_x, shape[1])

def grid_3(shape):
    blocks = shape[0] * shape[1] * shape[2]
    
    if blocks >= MINIMUM_BLOCK_SIZE:
        return shape
    
    block_x = shape[0] + ceil(MINIMUM_BLOCK_SIZE - blocks, shape[1] * shape[2])
    
    return (block_x, shape[1], shape[2])

def grid_config(shape):
    if len(shape) == 1:
        return grid_1(shape)
    elif len(shape) == 2:
        return grid_2(shape)
    else:
        return grid_3(shape)
