import numpy as np

def relu(signals: np.ndarray) -> np.ndarray:
    """evaluates the Leaky ReLU

      ReLU = {
        x * ALPHA, if x < 0
        x, otherwise
      }

    Args:
        signals (np.ndarray): [batch][1][N]

    Returns:
        np.ndarray: [batch][1][N]
    """
    ALPHA = 0.01
    return np.where(signals > 0, signals, signals * ALPHA)
