import numpy as np

def stochastic_gradient_descent(gradients: np.ndarray) -> np.ndarray:
    """sgd for gradients

    Args:
        gradients (np.ndarray): [BATCH][N][M]

    Returns:
        np.ndarray: [N][M]
    """
    
    BATCH = gradients.shape[0]
    N = gradients.shape[1]
    M = gradients.shape[2]
    
    ret = np.empty(shape=(N, M), dtype=np.float64)
    
    for i in range(N):
        for j in range(M):
            acm = 0.0
            
            for k in range(BATCH):
                acm += gradients[k, i, j]
            
            ret[i, j] = acm
    
    return ret
