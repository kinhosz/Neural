import numpy as np

def sigmoid2_derivate(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates the derivate of sigmoid2:
    
    f'(x) = 2*exp(-x)/(1 + exp(-x))^2

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    
    ret = []
    
    for signal, alpha in zip(signals, alphas):
        exp_signal = np.exp(-signal)
        exp_plus_one_square = (exp_signal + 1) * (exp_signal + 1)
        derivate = (2.0 * exp_signal) / exp_plus_one_square
        
        ret.append(
            derivate * alpha
        )
    
    return np.array(ret)
