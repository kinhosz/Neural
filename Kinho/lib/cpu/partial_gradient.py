import numpy as np

def partial_gradient(weigth: np.ndarray, eta: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """update: weight = weight - eta * gradient

    Args:
        weigth (np.ndarray): [N][M]
        eta (np.ndarray): [1]
        gradient (np.ndarray): [N][M]

    Returns:
        np.ndarray: weight
    """
    
    for i in range(weigth.shape[0]):
        for j in range(weigth.shape[1]):
            weigth[i, j] = weigth[i, j] - eta[0] * gradient[i, j]
    
    return weigth
