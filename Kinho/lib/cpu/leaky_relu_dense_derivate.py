import numpy as np

from .leaky_relu_dense import LEAK


def leaky_relu_dense_derivate(signals: np.ndarray, alphas: np.ndarray, leak: float = LEAK) -> np.ndarray:
    """evaluates the derivate of leaky_relu_dense:

    f'(x) = 1 if x > 0 else leak

    Args:
        signals (np.ndarray): [BATCH][1][N], the cached forward input
        alphas (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    return np.where(signals > 0, alphas, leak * alphas)
