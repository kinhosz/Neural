import numpy as np

def relu_derivate(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates the derivate of leaky relu
    
    f'(x) = {
        ALPHA, if x < 0
        1.0, otherwise
    }

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    ALPHA = 0.01
    nabla = np.where(signals > 0, 1.0, ALPHA)
    
    return nabla * alphas
