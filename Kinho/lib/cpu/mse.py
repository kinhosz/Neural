import numpy as np

def mse(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """mse error for predict and target

    Args:
        predict (np.ndarray): [BATCH][1][N]
        target (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [1]
    """
    
    diff = predict - target
    diff_square = diff * diff
    ret = diff_square.sum() / target.shape[2]
    
    return np.array([ret])
