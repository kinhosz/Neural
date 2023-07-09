import numpy as np

def transpose(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates the gradient for a matrix_multiplication
    using signals as input and alphas as error

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][M]

    Returns:
        np.ndarray: [BATCH][N][M]
    """
    
    ret = []
    
    for signal, alpha in zip(signals, alphas):
        ret.append(
            signal.transpose().dot(alpha)
        )
    
    return np.array(ret)