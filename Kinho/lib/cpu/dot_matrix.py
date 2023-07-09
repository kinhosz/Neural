import numpy as np

def dot_matrix(
    signals: np.ndarray, weight: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """
    Args:
        signals (np.array): the batch input for network -> shape(minibatch, 1, dim)
        weight (np.array): the weights of a single layer -> shape(dim, dim2)
        bias (np.array): the biase of a single layer -> (1, dim2)

    Returns:
        np.array: the output batch for layer -> shape(minibatch, 1, dim2)
    """
    ret = []

    for signal in signals:
        ret.append(signal.dot(weight) + bias)

    return np.array(ret)
