import numpy as np

def bce(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """BCE - Binary Cross Entropy

        BCE = (-1.0 / BATCH) * SUM(1<=i<=BATCH)[Y[i] * log(P[i]) + (1.0 - Y[i]) * log(1.0 - P[i])]

        where:
            Y[i] = target[i]
            P[i] = predict[i]

    Args:
        predict (np.ndarray): [BATCH][1][N]
        target (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [1]
    """

    res = (target * np.log(predict)) + ((1.0 - target) * np.log(1.0 - predict))
    val = (-1.0 / (predict.shape[0] + predict.shape[2])) * res.sum()

    return np.array([val])
