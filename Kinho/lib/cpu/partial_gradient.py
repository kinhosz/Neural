import numpy as np

def partial_gradient(weight: np.ndarray, eta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """update: weight = weight - eta * gradient

    Args:
        weigth (np.ndarray): [N][M]
        eta (np.ndarray): [1]
        gradient (np.ndarray): [N][M]

    Returns:
        np.ndarray: weight
    """
    
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight[i, j] = weight[i, j] - eta[0] * gradient[i, j]
    
    return weight
