import numpy as np

def sigmoid2_derivate(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates the derivate of sigmoid2:
    
    f'(x) = exp(-x)/(1 + exp(-x))^2
    f'(x) = 1/(e^x + 2 + e^(-x)) simplified

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    
    ret = []
    
    for signal, alpha in zip(signals, alphas):
        exp_minus_signal = np.exp(-signal)
        exp_signal = np.exp(signal)
        sum_total = (exp_signal + exp_minus_signal + 2.0)
        derivate = 1.0 / sum_total
        
        ret.append(
            derivate * alpha
        )
    
    return np.array(ret)
