import numpy as np

def sigmoid2(signals: np.ndarray) -> np.ndarray:
    """evaluates : 1/(1 + e^(-z))
    all values are mapped into a range (0.0, 1.0)

    Args:
        signals (np.ndarray): [batch][1][N]

    Returns:
        np.ndarray: [batch][1][N]
    """
    ret = []
    
    for signal in signals:
        ret.append(
            (1.0 / (1.0 + np.exp(-signal)))
        )
    
    return np.array(ret)
