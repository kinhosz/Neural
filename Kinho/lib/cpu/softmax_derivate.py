import numpy as np

def softmax_derivate(signals: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """evaluates: the derivate of softmax

    Args:
        signals (np.ndarray): [BATCH][1][N]
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    
    ret = []
    
    for signal, alpha in zip(signals, alphas):
        exp_signal = np.exp(signal)
        exp_signal_sum = exp_signal.sum()
        exp_times_sum = exp_signal * exp_signal_sum
        square_exp = exp_signal * exp_signal
        derivate = (exp_times_sum - square_exp)/(exp_signal_sum * exp_signal_sum)
        
        ret.append(derivate * alpha)
    
    return np.array(ret)
