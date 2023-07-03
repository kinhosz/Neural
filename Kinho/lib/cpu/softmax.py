import numpy as np

def softmax(signals: np.ndarray) -> np.ndarray:
    """evaluates e^x/sum{e^x}

    Args:
        signals (np.ndarray): the signal sent

    Returns:
        np.ndarray: the evaluated signal
    """
    ret = []
    
    for signal in signals:
        z = np.exp(signal)
        sumT = z.sum()
        z = z/sumT
        ret.append(z)
    
    return np.array(ret)
