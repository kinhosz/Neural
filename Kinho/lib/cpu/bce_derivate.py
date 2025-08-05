import numpy as np

def bce_derivate(predict: np.ndarray, target: np.ndarray) -> np.ndarray:
    """BCE - Binary Cross Entropy Derivate

        BCE = (1.0 / BATCH) * SUM(1<=i<=BATCH)[ (P[i] - Y[i]) / (P[i] * (1.0 - P[i])) ]

        where:
            Y[i] = target[i]
            P[i] = predict[i]

    Args:
        predict (np.ndarray): [BATCH][1][N]
        target (np.ndarray): [BATCH][1][N]

    Returns:
        np.ndarray: [BATCH][1][N]
    """
    BATCH = predict.shape[0]

    res = (predict - target) / (predict * (1.0 - predict))
    res = res / BATCH

    return res
