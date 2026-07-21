import numpy as np

LEAK = 0.01


def leaky_relu_dense(signals: np.ndarray, leak: float = LEAK) -> np.ndarray:
    """evaluates: x if x > 0 else leak * x, for the dense head's batched signals

    Args:
        signals (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    return np.where(signals > 0, signals, leak * signals)
